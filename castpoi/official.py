"""Load the official STHGCN/LLM4POI splits.

`data_official/<ds>/` holds the output of LLM4POI's own preprocessing, run
unmodified; see DATA.md. The regenerated `train_sample.csv` is byte-identical to
the published `w11wo/LLM4POI` files for all three datasets, apart from 424
mojibake rows in that upload.

The properties below come from the official pipeline, not from this module:
min_poi_freq / min_user_freq of 9 / 9 with `count > freq` semantics (CA is
filtered twice; NYC arrives pre-split from GETNext), a global chronological
80/10/10 split over all check-ins, 24-hour session gaps with singleton sessions
dropped, and removal of val/test rows whose user or POI is absent from train.

Two things are done here because the official files leave them:

1. `UTCTimeOffsetEpoch` is corrupt in every published file (see timeparse.py), so
   `UTCTimeOffset` is parsed instead.
2. POI id 0 is a real POI in their label encoding, while the model reserves 0 for
   padding, so ids are shifted where needed.
"""
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .timeparse import TIME_SPEC, assert_human_rhythm, parse_times

DEFAULT_OFFICIAL = Path(__file__).resolve().parent.parent / "data_official"

FILES = {
    "train": "train_sample.csv",
    "val": "validate_sample_with_traj.csv",
    "test": "test_sample_with_traj.csv",
}

SOURCE = {
    "nyc": {"note": "Foursquare New York City", "platform": "Foursquare",
            "provenance": "GETNext pre-split NYC_{train,val,test}.csv"},
    "tky": {"note": "Foursquare Tokyo", "platform": "Foursquare",
            "provenance": "STHGCN filter(9,9) + global chronological 80/10/10"},
    "ca": {"note": "Gowalla California", "platform": "Gowalla",
           "provenance": "STHGCN filter(9,9) applied TWICE + global chronological 80/10/10"},
}

# The official pipeline label-encodes ids under two conventions: NYC yields
# ids 0..N-1 with padding N, while TKY and CA yield ids 1..N with padding 0. So
# PoiId==0 is a real POI in NYC and the padding bucket in TKY/CA. Treating them
# alike would count the padding bucket as an entity and, given that 0 is reserved
# for sequence padding here, would leave a dead embedding row in TKY/CA. The
# convention is declared explicitly rather than inferred from min(), so a split
# that happens not to contain id 0 cannot be misread.
ID_CONVENTION = {"nyc": "zero_indexed", "tky": "one_indexed", "ca": "one_indexed"}


class OfficialDataMissing(RuntimeError):
    pass


def _read(ds: str, root: Path) -> Dict[str, pd.DataFrame]:
    d = root / ds
    out = {}
    for split, fn in FILES.items():
        p = d / fn
        if not p.exists():
            raise OfficialDataMissing(
                f"{p} not found. Regenerate with LLM4POI's own pipeline; see "
                f"DATA.md. This code will not "
                f"substitute its own preprocessing.")
        out[split] = pd.read_csv(p, low_memory=False)
    return out


def poi_shift(ds: str) -> int:
    """Offset mapping the official PoiId onto our vocabulary, where 0 = padding.

    zero_indexed (nyc): official ids 0..N-1 -> 1..N, shift +1.
    one_indexed  (tky, ca): official ids are already 1..N with 0 as their own
                            padding bucket -> no shift.
    """
    return 1 if ID_CONVENTION[ds] == "zero_indexed" else 0


def real_id_range(ds: str, train: pd.DataFrame, col: str) -> Tuple[int, int]:
    """(n_distinct_real, our_max_index) for a label-encoded column."""
    if ID_CONVENTION[ds] == "zero_indexed":
        n = int(train[col].max()) + 1        # 0..max are all real; padding is max+1
    else:
        n = int(train[col].max())            # 1..max are real; 0 is padding
    return n, n


def _trajectories(df: pd.DataFrame, ds: str) -> Dict[int, List[Dict]]:
    df = parse_times(df, ds)
    # mergesort = stable, so exact (UserId, ts_utc) ties keep their file order
    # instead of pandas' quicksort tie-break. TKY ships 467 exact duplicate
    # (UserId, ts_utc, PoiId) rows and CA 35; with an unstable sort, a duplicate
    # of the target could land in the *history* of its own eval sample. Only 36
    # samples out of 60k were affected, so this changes no conclusion, but it is
    # the difference between "causal" and "causal except when pandas feels like it".
    df = df.sort_values(["UserId", "ts_utc"], kind="mergesort")
    has_cat = "PoiCategoryName" in df.columns
    # check_ins_id rides along unused by the model. It is the only stable join key
    # back to a foreign implementation's test set: the official pipeline assigns it
    # (preprocess_main.py:45) as a rank over UTCTimeOffset, a wall-clock string, so
    # it survives the timezone corruption that makes UTCTimeOffsetEpoch unusable
    # across machines. Without it, comparing against a re-run of STHGCN means
    # replaying this function's grouping from outside and hoping it stays in sync.
    cols = ["UserId", "PoiId", "ts_utc", "local_hour", "local_dow",
            "Latitude", "Longitude", "pseudo_session_trajectory_id", "check_ins_id"]
    if has_cat:
        cols.append("PoiCategoryName")

    shift = poi_shift(ds)
    trajs: Dict[int, List[Dict]] = {}
    for row in df[cols].itertuples(index=False, name=None):
        trajs.setdefault(int(row[0]), []).append({
            "poi_idx": int(row[1]) + shift,      # 0 reserved for padding
            "ts_utc": float(row[2]),
            "hour": float(row[3]),
            "dow": int(row[4]),
            "latitude": float(row[5]),
            "longitude": float(row[6]),
            "traj_id": int(row[7]),
            "check_ins_id": int(row[8]),
            "category": row[9] if has_cat else "Unknown",
        })
    return trajs


def fingerprint(splits: Dict[str, Dict], num_pois: int) -> str:
    h = hashlib.sha256()
    h.update(f"OFFICIAL_V1|{num_pois}".encode())
    for name in ("train", "val", "test"):
        h.update(f"|{name}|".encode())
        for uid in sorted(splits[name]):
            h.update(f"{uid}:".encode())
            h.update(np.array([c["poi_idx"] for c in splits[name][uid]], dtype=np.int64).tobytes())
            h.update(np.array([c["ts_utc"] for c in splits[name][uid]], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def load_official(ds: str, root: Path = None) -> Dict:
    ds = ds.lower()
    root = Path(root or DEFAULT_OFFICIAL)
    info = SOURCE[ds]
    print(f"\n{'=' * 68}\n[official] {ds.upper()} ({info['note']}, {info['platform']})\n"
          f"[official] provenance: {info['provenance']}\n{'=' * 68}")

    raw = _read(ds, root)
    for k, df in raw.items():
        print(f"[official] {k:5s}: {len(df):>7,} check-ins  "
              f"({(root / ds / FILES[k]).stat().st_size / 1e6:.1f} MB)")

    # Vocabulary comes from TRAIN, exactly as their id_encode does. val/test have
    # already had unseen users and POIs removed upstream.
    shift = poi_shift(ds)
    n_train_pois, max_idx = real_id_range(ds, raw["train"], "PoiId")
    n_users, _ = real_id_range(ds, raw["train"], "UserId")
    num_pois = max_idx + 1                           # index 0 is our padding

    for k in ("val", "test"):
        lo, hi = raw[k]["PoiId"].min() + shift, raw[k]["PoiId"].max() + shift
        assert 1 <= lo and hi <= max_idx, (
            f"{ds}/{k} PoiId maps outside 1..{max_idx} (got {lo}..{hi}); "
            f"unseen POIs should have been removed upstream")

    splits = {k: _trajectories(df, ds) for k, df in raw.items()}
    used = {c["poi_idx"] for s in splits.values() for t in s.values() for c in t}
    assert 0 not in used, f"{ds}: padding index 0 leaked into the data"
    dead = set(range(1, num_pois)) - used
    if dead:
        print(f"[official] note: {len(dead)} vocabulary slots never appear in any split")

    all_df = parse_times(pd.concat(raw.values()), ds)
    st = assert_human_rhythm(all_df["local_hour"].values, ds)

    # POI metadata is aggregated over the train split only, so no statistic is
    # computed over test. POIs absent from train keep coordinate (0,0) and
    # category Unknown; their embeddings never receive a gradient.
    train_df = all_df[all_df["_split"] == "train"] if "_split" in all_df.columns else \
        parse_times(raw["train"], ds)
    poi_locations = np.zeros((num_pois, 2), dtype=np.float64)
    poi_categories: Dict[int, str] = {}
    agg = {"Latitude": "mean", "Longitude": "mean"}
    if "PoiCategoryName" in train_df.columns:
        agg["PoiCategoryName"] = lambda x: x.mode().iloc[0] if len(x.mode()) else "Unknown"
    for pid, row in train_df.groupby("PoiId").agg(agg).iterrows():
        i = int(pid) + shift
        if not 1 <= i < num_pois:
            continue                                 # the official padding bucket
        poi_locations[i] = [row["Latitude"], row["Longitude"]]
        poi_categories[i] = row.get("PoiCategoryName", "Unknown")

    counts = np.zeros(num_pois, dtype=np.float64)
    for traj in splits["train"].values():
        for c in traj:
            counts[c["poi_idx"]] += 1
    counts[0] = 0.0
    popularity = counts / counts.sum()

    n_check = sum(len(df) for df in raw.values())
    n_traj = int(pd.concat(raw.values())["pseudo_session_trajectory_id"].nunique())
    stats = {
        "dataset": ds.upper(), "platform": info["platform"], "note": info["note"],
        "provenance": info["provenance"], "id_convention": ID_CONVENTION[ds],
        "users": n_users, "pois": n_train_pois, "checkins": n_check,
        "trajectories": n_traj,
        "sparsity_pct": 100 * (1 - n_check / (n_users * n_train_pois)),
        "avg_traj_len": n_check / n_traj,
        "train_checkins": len(raw["train"]), "val_checkins": len(raw["val"]),
        "test_checkins": len(raw["test"]),
        "train_users": len(splits["train"]), "test_users": len(splits["test"]),
        "tz": TIME_SPEC[ds]["tz"], "time_column_means": TIME_SPEC[ds]["column_means"],
        "rhythm_trough_hour": st["argmin_hour"], "rhythm_day_night_ratio": st["day_night_ratio"],
        "num_pois_incl_pad": num_pois,
        "data_fingerprint": fingerprint(splits, num_pois),
    }
    print(f"[official] users={n_users:,} POIs={n_train_pois:,} check-ins={n_check:,} "
          f"trajectories={n_traj:,}")
    print(f"[official] local time OK (trough {st['argmin_hour']:02d}:00, "
          f"day/night {st['day_night_ratio']:.1f}) via {TIME_SPEC[ds]['tz']}, "
          f"column read as {TIME_SPEC[ds]['column_means']}")
    print(f"[official] fingerprint {stats['data_fingerprint']}")

    # Time origin. The model only ever uses DIFFERENCES of timestamps, so a
    # constant offset cancels exactly; measuring hours from the dataset's own
    # start instead of from 1970 keeps the magnitude near 1e4 rather than 1e9.
    # That is what lets the tensors be float32 without reintroducing the 128 s
    # quantization bug (float32 ulp at 1.3e9 s is 128 s; at 1.3e4 h it is 3.5 s).
    # It also makes the code run on MPS, which cannot hold float64 at all.
    t_ref = min(c["ts_utc"] for s_ in splits.values() for t in s_.values() for c in t)

    return {
        "t_ref": float(t_ref),
        "train_data": splits["train"], "val_data": splits["val"], "test_data": splits["test"],
        "poi_locations": poi_locations, "poi_categories": poi_categories,
        "poi_popularity": popularity, "num_pois": num_pois, "stats": stats,
        "default_lat": float(train_df["Latitude"].mean()),
        "default_lon": float(train_df["Longitude"].mean()),
    }
