"""Build a per-user leave-one-out split from the chronological official data.

For each user, ordered by time: the last check-in is test, the second-to-last is
validation, the rest are train. Users with fewer than 3 check-ins are dropped, and
val/test rows whose POI or user is unseen in the new train are removed. Output keeps
the original columns so the loader reads it unchanged.

    python build_loo_split.py --src data_official --out data_loo --datasets nyc tky ca
"""
import argparse
import os
import pandas as pd

FILES = {"train": "train_sample.csv",
         "val":   "validate_sample_with_traj.csv",
         "test":  "test_sample_with_traj.csv"}


def build(ds, src, out):
    parts = [pd.read_csv(os.path.join(src, ds, fn), low_memory=False) for fn in FILES.values()]
    df = pd.concat(parts, ignore_index=True)

    df["_t"] = pd.to_datetime(df["UTCTimeOffset"].astype(str).str.slice(0, 19),
                              format="%Y-%m-%d %H:%M:%S", errors="coerce")
    assert df["_t"].notna().all(), f"{ds}: some UTCTimeOffset failed to parse"
    df = df.sort_values(["UserId", "_t"], kind="mergesort").reset_index(drop=True)

    df["_rk"] = df.groupby("UserId").cumcount(ascending=False)
    df["_n"] = df.groupby("UserId")["UserId"].transform("size")
    df = df[df["_n"] >= 3].copy()
    df["_split"] = "train"
    df.loc[df["_rk"] == 0, "_split"] = "test"
    df.loc[df["_rk"] == 1, "_split"] = "val"

    train_pois = set(df.loc[df["_split"] == "train", "PoiId"].unique())
    train_users = set(df.loc[df["_split"] == "train", "UserId"].unique())
    is_train = df["_split"] == "train"
    seen = df["PoiId"].isin(train_pois) & df["UserId"].isin(train_users)
    df = df[is_train | seen].copy()

    orig_cols = [c for c in df.columns if not c.startswith("_")]
    os.makedirs(os.path.join(out, ds), exist_ok=True)
    counts = {}
    for split, fn in FILES.items():
        sub = df.loc[df["_split"] == split, orig_cols]
        sub.to_csv(os.path.join(out, ds, fn), index=False)
        counts[split] = len(sub)
    print(f"[{ds}] train {counts['train']:,} / val {counts['val']:,} / test {counts['test']:,} "
          f"| users {df['UserId'].nunique():,} | POIs {len(train_pois):,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data_official")
    ap.add_argument("--out", default="data_loo")
    ap.add_argument("--datasets", nargs="+", default=["nyc", "tky", "ca"])
    a = ap.parse_args()
    for ds in a.datasets:
        build(ds, a.src, a.out)


if __name__ == "__main__":
    main()
