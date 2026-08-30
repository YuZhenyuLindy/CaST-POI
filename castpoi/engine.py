"""Training and evaluation loops.

Per-epoch history, per-sample test ranks and timing are written to disk so that
every reported number can be recomputed from a file rather than from memory.
"""
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .metrics import metrics_from_ranks, per_sample_ranks, summarize, format_metrics


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, eta_min: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            factor = self.epoch / max(self.warmup_epochs, 1)
        else:
            progress = (self.epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            factor = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = self.eta_min + (base - self.eta_min) * factor

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def _query_spatial_context(batch: Dict[str, torch.Tensor]):
    """Current position and previous position, both taken from the history.

    An earlier version passed `target_location`, the coordinates of the POI being
    predicted. SpatialEncoding turns that into the displacement to the answer,
    which is future information; results produced before protocol_rev 2 carry
    that leak. Sequences are left-padded, so column -1 is the most recent real
    check-in and -2 the one before it.

    A user with a single check-in of history has no column -2, only padding filled
    with the dataset mean coordinate. Those users get prev := current, i.e. zero
    displacement, rather than a fictitious move from the centroid.
    """
    cur = batch["locations"][:, -1, :]
    prev = batch["locations"][:, -2, :]
    has_prev = (batch["seq_lengths"] >= 2).unsqueeze(-1).to(cur.dtype)
    return cur, prev * has_prev + cur * (1 - has_prev)


def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _forward(model, batch):
    return model(
        poi_ids=batch["poi_ids"],
        ts_hours=batch["ts_hours"],
        hour=batch["hour"],
        dow=batch["dow"],
        locations=batch["locations"],
        seq_lengths=batch["seq_lengths"],
        query_hour=batch["target_hour"],
        query_dow=batch["target_dow"],
        query_location=_query_spatial_context(batch)[0],
        prev_query_location=_query_spatial_context(batch)[1],
        repeat_hist=batch.get("repeat_hist"),
    )


def train_epoch(model, loader, optimizer, criterion, device, config) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = _to_device(batch, device)
        h_proj, _, rep = _forward(model, batch)
        if config.get("train_objective", "sampled") == "full":
            # Same scoring path evaluation uses, so training and testing now pose
            # the model the identical |L|-way problem.
            logits = model.compute_all_scores(h_proj, rep)
            loss = criterion.forward_full(logits, batch["target_poi"], batch.get("is_explore"))
        else:
            pos, neg = model.compute_sampled_scores(h_proj, batch["target_poi"], batch["neg_ids"], rep)
            loss = criterion(pos, neg, batch.get("is_explore"))
        if not torch.isfinite(loss):
            raise RuntimeError(
                "training loss became NaN/Inf. Refusing to continue: a diverged model "
                "scores NaN, and NaN ranks as position 1 for every sample, which would "
                "be reported as a flawless 100% HR@k. Lower the learning rate, or check "
                "the device (Apple MPS has produced NaN here where CPU and CUDA do not).")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config["gradient_clip"] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


TOPK_KEEP = 20


@torch.no_grad()
def evaluate(model, loader, device, ks=(5, 10, 20), collect_alpha: bool = False,
             collect_topk: bool = False):
    """Returns (summary, per_sample_dict, extras). per_sample arrays enable paired tests."""
    model.eval()
    ranks, alphas, targets, topk = [], [], [], []
    for batch in loader:
        batch = _to_device(batch, device)
        h_proj, alpha, rep = _forward(model, batch)
        scores = model.compute_all_scores(h_proj, rep)
        ranks.append(per_sample_ranks(scores, batch["target_poi"]).cpu().numpy())
        targets.append(batch["target_poi"].cpu().numpy())
        # Rank-based metrics are recoverable from `ranks` alone, but coverage,
        # novelty and popularity bias need the predicted ids. Off by default
        # because validation runs this every epoch and discards the result;
        # run.py enables it for the single test evaluation that is saved.
        if collect_topk:
            topk.append(scores.topk(min(TOPK_KEEP, scores.size(1)),
                                    dim=1).indices.cpu().numpy().astype(np.int32))
        if collect_alpha:
            alphas.append(alpha.cpu().numpy())

    ranks = np.concatenate(ranks).astype(np.float64)
    per_sample = metrics_from_ranks(ranks, ks)
    extras = {"ranks": ranks, "targets": np.concatenate(targets),
              "topk": np.concatenate(topk) if topk else None}

    # Official check_ins_id per sample, so these ranks can be joined to a foreign
    # implementation's. Position i of `ranks` means sample i of the dataset only
    # because the eval loaders are built with shuffle=False; under a shuffling
    # sampler the join would still run and would pair every sample with the wrong
    # id, so check rather than assume.
    ds = getattr(loader, "dataset", None)
    if hasattr(ds, "check_ins_ids"):
        if not isinstance(loader.sampler, torch.utils.data.SequentialSampler):
            raise RuntimeError(
                f"eval loader uses {type(loader.sampler).__name__}, not SequentialSampler. "
                f"Rank i would not correspond to sample i, so check_ins_id would mislabel "
                f"every row and any paired test built on it would be silently wrong.")
        cids = ds.check_ins_ids
        if len(cids) != len(ranks):
            raise RuntimeError(f"{len(cids)} check_ins_ids vs {len(ranks)} ranks.")
        extras["check_ins_id"] = cids
    if collect_alpha:
        extras["alpha"] = np.concatenate(alphas, axis=0)
    return summarize(per_sample), per_sample, extras


def train_model(model, train_loader, val_loader, config, device, logger=None) -> Tuple[nn.Module, Dict]:
    """Train with warmup+cosine LR and early stopping on validation HR@10."""
    from .layers import CELoss

    log = (logger.info if logger else print)
    model = model.to(device)
    criterion = CELoss(config["label_smoothing"], config["explore_weight"],
                           config.get("bpr_weight", 0.5), config.get("bpr_margin", 1.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = WarmupCosineScheduler(optimizer, config["warmup_epochs"], config["num_epochs"])

    history = {"epochs": [], "best_epoch": None, "stopped_early": False}
    best_hr10, best_state, patience = -1.0, None, 0
    t_start = time.time()

    for epoch in range(1, config["num_epochs"] + 1):
        t0 = time.time()
        loss = train_epoch(model, train_loader, optimizer, criterion, device, config)
        train_s = time.time() - t0
        scheduler.step()

        t1 = time.time()
        val_metrics, _, _ = evaluate(model, val_loader, device, config["eval_ks"])
        eval_s = time.time() - t1

        history["epochs"].append({
            "epoch": epoch, "train_loss": loss, "lr": scheduler.get_lr(),
            "train_seconds": train_s, "eval_seconds": eval_s,
            "val": val_metrics,
        })
        log(f"epoch {epoch:3d} | loss {loss:.4f} | lr {scheduler.get_lr():.2e} | "
            f"{train_s:.1f}s+{eval_s:.1f}s | val {format_metrics(val_metrics)}")

        improved = val_metrics["HR@10"] > best_hr10
        if improved:
            best_hr10 = val_metrics["HR@10"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            patience = 0
        elif epoch > config["warmup_epochs"]:
            patience += 1
            if patience >= config["early_stopping_patience"]:
                log(f"early stop at epoch {epoch} (best val HR@10 {best_hr10 * 100:.2f} @ epoch {history['best_epoch']})")
                history["stopped_early"] = True
                break

    history["total_train_seconds"] = time.time() - t_start
    history["best_val_hr10"] = best_hr10
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def measure_inference(model, loader, device, batch_size: int, n_warmup: int = 3, n_iters: int = 20) -> Dict:
    """Honest latency/throughput: measured separately, not derived one from the other."""
    model.eval()
    batch = _to_device(next(iter(loader)), device)
    B = batch["poi_ids"].size(0)

    for _ in range(n_warmup):
        h, _, rep = _forward(model, batch)
        model.compute_all_scores(h, rep)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        h, _, rep = _forward(model, batch)
        model.compute_all_scores(h, rep)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    per_batch_ms = float(times.mean() * 1000)
    out = {
        "batch_size": B,
        "batch_latency_ms_mean": per_batch_ms,
        "batch_latency_ms_std": float(times.std(ddof=1) * 1000) if len(times) > 1 else 0.0,
        "per_query_latency_ms": per_batch_ms / B,
        "throughput_queries_per_s": B / (per_batch_ms / 1000),
        "n_iters": n_iters,
        "device": str(device),
    }
    if device.type == "cuda":
        out["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    return out


