"""Torch datasets and dataloaders.

No preprocessing happens here. Filtering and splitting come from the official
files loaded by official.py and from build_loo_split.py; see DATA.md. This module
only turns those trajectories into batched tensors.
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DataUnavailable(RuntimeError):
    """Kept for API compatibility. Real data errors now raise OfficialDataMissing."""


def _pack_repeat_history(history: List[Dict], max_len: int) -> List[int]:
    """POI ids over a long window, used only by the revisit features.

    Kept separate from the sequence window: attention is O(L) per candidate, so
    the reader truncates to max_history_len, but counting visits is cheap and a
    longer window classifies more targets correctly as revisits.
    """
    return [c["poi_idx"] for c in history[-max_len:]]


def _pack(history: List[Dict], max_len: int, default_lat: float, default_lon: float,
          t_ref: float = 0.0):
    """Truncate to the last max_len, then LEFT-pad to a fixed width.

    Timestamps come out as HOURS SINCE t_ref, not raw epoch seconds. See
    official.load_official for why: differences are all the model uses, and the
    smaller magnitude is what makes float32 safe (and MPS possible)."""
    h = history[-max_len:]
    n = len(h)
    pad = max_len - n
    poi = [0] * pad + [c["poi_idx"] for c in h]
    ts = [(h[0]["ts_utc"] - t_ref) / 3600.0] * pad + [(c["ts_utc"] - t_ref) / 3600.0 for c in h]
    hour = [0.0] * pad + [c["hour"] for c in h]
    dow = [0] * pad + [c["dow"] for c in h]
    loc = [[default_lat, default_lon]] * pad + [[c["latitude"], c["longitude"]] for c in h]
    return poi, ts, hour, dow, loc, n


def _tensors(poi, ts, hour, dow, loc, n, target, t_ref=0.0, rep_hist=None, rep_len=0):
    d = {
        "poi_ids": torch.tensor(poi, dtype=torch.long),
        "ts_hours": torch.tensor(ts, dtype=torch.float32),   # hours since t_ref
        "hour": torch.tensor(hour, dtype=torch.float32),
        "dow": torch.tensor(dow, dtype=torch.long),
        "locations": torch.tensor(loc, dtype=torch.float32),
        "seq_len": torch.tensor(n, dtype=torch.long),
        "target_poi": torch.tensor(target["poi_idx"], dtype=torch.long),
        "target_hour": torch.tensor(target["hour"], dtype=torch.float32),
        "target_dow": torch.tensor(target["dow"], dtype=torch.long),
        "target_location": torch.tensor([target["latitude"], target["longitude"]], dtype=torch.float32),
    }
    if rep_hist is not None:
        pad = rep_len - len(rep_hist)
        d["repeat_hist"] = torch.tensor([0] * pad + rep_hist, dtype=torch.long)
    return d


class POITrainDataset(Dataset):
    def __init__(self, train_data, num_pois, poi_popularity, config,
                 default_lat=0.0, default_lon=0.0, t_ref=0.0):
        self.num_pois = num_pois
        self.t_ref = t_ref
        # Under train_objective="full" nothing consumes neg_ids, and sampling
        # them anyway would cost ~200 ms per batch of 512 for a tensor the
        # training step throws away.
        self.sample_negatives = config.get("train_objective", "sampled") != "full"
        self.num_negatives = config["num_negatives"]
        self.max_history_len = config["max_history_len"]
        self.repeat_history_len = config.get("repeat_history_len", 512)
        self.default_lat, self.default_lon = default_lat, default_lon
        self.valid_idx = np.where(poi_popularity > 0)[0]
        p = poi_popularity[self.valid_idx]
        self.valid_probs = p / p.sum()
        # np.random.choice(p=...) rebuilds an O(|V|) cumulative sum on EVERY call.
        # At 499 negatives x ~82k samples per epoch that dominated the whole
        # training step: measured 38 s/epoch of pure data loading on a fast CPU,
        # which on a 2-vCPU Colab runtime left the GPU idle ~75% of the time.
        # Build the CDF once; sample with searchsorted, same distribution.
        self._cdf = np.cumsum(self.valid_probs)
        self._cdf[-1] = 1.0

        self.samples = []
        for uid, traj in train_data.items():
            seen = set()
            for i in range(1, len(traj)):
                seen.add(traj[i - 1]["poi_idx"])
                self.samples.append({"history": traj[:i], "target": traj[i],
                                     "is_explore": traj[i]["poi_idx"] not in seen})

    def __len__(self):
        return len(self.samples)

    def _sample_popularity(self, size: int) -> np.ndarray:
        """Popularity-weighted draw via the prebuilt CDF. Same distribution as
        np.random.choice(p=self.valid_probs), without its per-call O(|V|) setup."""
        return self.valid_idx[np.searchsorted(self._cdf, np.random.random(size))]

    def _sample_negatives(self, target: int) -> List[int]:
        """Half popularity-weighted, half uniform, de-duplicated, target excluded.

        Plain Python with a set: at these sizes (~500 draws over a 5k vocabulary)
        a vectorised numpy version measured slower. Costs roughly 200 ms per batch
        of 512, which num_workers > 0 hides.
        """
        n_pop = self.num_negatives // 2
        neg = set()
        for _ in range(8):
            if len(neg) >= n_pop:
                break
            neg.update(int(c) for c in self._sample_popularity(n_pop * 2) if c != target)
            if len(neg) > n_pop:
                neg = set(list(neg)[:n_pop])
        for _ in range(8):
            if len(neg) >= self.num_negatives:
                break
            for c in np.random.randint(1, self.num_pois, size=(self.num_negatives - len(neg)) * 2):
                if c != target:
                    neg.add(int(c))
                if len(neg) >= self.num_negatives:
                    break
        out = list(neg)[: self.num_negatives]
        while len(out) < self.num_negatives:
            r = int(np.random.randint(1, self.num_pois))
            if r != target:
                out.append(r)
        return out

    def __getitem__(self, idx):
        s = self.samples[idx]
        packed = _pack(s["history"], self.max_history_len, self.default_lat,
                       self.default_lon, self.t_ref)
        d = _tensors(*packed, s["target"], self.t_ref,
                     _pack_repeat_history(s["history"], self.repeat_history_len),
                     self.repeat_history_len)
        if self.sample_negatives:
            d["neg_ids"] = torch.tensor(self._sample_negatives(s["target"]["poi_idx"]), dtype=torch.long)
        d["is_explore"] = torch.tensor(float(s["is_explore"]), dtype=torch.float32)
        return d


class POIEvalDataset(Dataset):
    def __init__(self, history_base: Dict, eval_data: Dict, config,
                 default_lat=0.0, default_lon=0.0, t_ref=0.0):
        self.max_history_len = config["max_history_len"]
        self.repeat_history_len = config.get("repeat_history_len", 512)
        self.t_ref = t_ref
        self.default_lat, self.default_lon = default_lat, default_lon
        self.samples = []
        for uid, traj in eval_data.items():
            base = history_base.get(uid, [])
            for i in range(len(traj)):
                history = base + traj[:i]
                if history:
                    self.samples.append({"history": history, "target": traj[i]})

    def __len__(self):
        return len(self.samples)

    @property
    def check_ins_ids(self) -> np.ndarray:
        """Official check_ins_id of each sample's target, in sample order.

        This is what lets a rank vector from this repo be joined to one from a
        foreign implementation of the same task.

        Sample order is a permutation of test-CSV row order in general: the loop
        above walks users, and any user whose first eval check-in has no prior
        history contributes no sample at all. On the official NYC files it happens
        to come out as the identity -- 9,074 samples, 9,074 rows, same sequence --
        because those files arrive sorted by (UserId, ts_utc) and every test user
        already has training history. That is a property of the data, not of this
        code, so do not index into the CSV by position; join on the id.

        Derived from self.samples rather than re-walking eval_data so it stays a
        projection of the thing it labels: if the loop above changes, this follows
        instead of quietly disagreeing with it.
        """
        return np.array([s["target"]["check_ins_id"] for s in self.samples], dtype=np.int64)

    def __getitem__(self, idx):
        s = self.samples[idx]
        packed = _pack(s["history"], self.max_history_len, self.default_lat,
                       self.default_lon, self.t_ref)
        return _tensors(*packed, s["target"], self.t_ref,
                        _pack_repeat_history(s["history"], self.repeat_history_len),
                        self.repeat_history_len)


def collate_fn(batch):
    keys = ["poi_ids", "ts_hours", "hour", "dow", "locations", "target_poi",
            "target_hour", "target_dow", "target_location"]
    if "repeat_hist" in batch[0]:
        keys = keys + ["repeat_hist"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["seq_lengths"] = torch.stack([b["seq_len"] for b in batch])
    for k in ("neg_ids", "is_explore"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def _worker_init(worker_id: int) -> None:
    """Give each dataloader worker its own numpy stream.

    torch seeds each worker's `torch` RNG but NOT numpy's. Our negative sampler
    is pure numpy, so without this every worker would draw the SAME negatives --
    a silent correctness bug that only appears once num_workers > 0.
    """
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)


def create_dataloaders(data: Dict, config: Dict, num_workers: int = 0):
    t_ref = data.get("t_ref", 0.0)
    train_ds = POITrainDataset(data["train_data"], data["num_pois"], data["poi_popularity"],
                               config, data["default_lat"], data["default_lon"], t_ref)
    val_ds = POIEvalDataset(data["train_data"], data["val_data"], config,
                            data["default_lat"], data["default_lon"], t_ref)
    train_plus_val = {uid: traj + data["val_data"].get(uid, [])
                      for uid, traj in data["train_data"].items()}
    test_ds = POIEvalDataset(train_plus_val, data["test_data"], config,
                             data["default_lat"], data["default_lon"], t_ref)

    ebs = config.get("eval_batch_size", config["batch_size"])
    mk = lambda ds, bs, sh: DataLoader(ds, batch_size=bs, shuffle=sh, collate_fn=collate_fn,
                                       num_workers=num_workers,
                                       worker_init_fn=_worker_init if num_workers else None,
                                       persistent_workers=num_workers > 0,
                                       pin_memory=torch.cuda.is_available())
    print(f"[data] samples: train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")
    return (mk(train_ds, config["batch_size"], True), mk(val_ds, ebs, False), mk(test_ds, ebs, False))
