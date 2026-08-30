"""Paired bootstrap significance test on HR@k between two models.

Aligns the two models sample-by-sample via check_ins_id, averages the hit indicator
across seeds per sample, then bootstraps over samples for the HR@k difference, 95% CI,
and a two-sided p-value.

    python paired_sig.py --a runs/castpoi_loo_lr0.001/nyc/full --b runs/loo/nyc/core_recbole --k 10
"""
import argparse, glob, os
import numpy as np


def load_side(path, k):
    files = sorted(glob.glob(os.path.join(path, "**", "ranks.npy"), recursive=True))
    if not files and os.path.exists(os.path.join(path, "ranks.npy")):
        files = [os.path.join(path, "ranks.npy")]
    if not files:
        raise FileNotFoundError(f"no ranks.npy under {path}")
    per_cid = {}
    for rf in files:
        cf = os.path.join(os.path.dirname(rf), "check_ins_id.npy")
        if not os.path.exists(cf):
            raise FileNotFoundError(
                f"{cf} not found. Both runs must save check_ins_id.npy; "
                f"without it the two rank vectors cannot be aligned.")
        ranks, cids = np.load(rf), np.load(cf)
        hit = (ranks <= k).astype(np.float64)
        for c, h in zip(cids.tolist(), hit.tolist()):
            per_cid.setdefault(c, []).append(h)
    return {c: float(np.mean(v)) for c, v in per_cid.items()}, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nboot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    args = ap.parse_args()

    A, na = load_side(args.a, args.k)
    B, nb = load_side(args.b, args.k)
    common = sorted(set(A) & set(B))
    if not common:
        raise SystemExit("no overlapping check_ins_id between the two runs")
    la, lb = args.labels
    a = np.array([A[c] for c in common]); b = np.array([B[c] for c in common])
    n = len(common)
    hr_a, hr_b = a.mean() * 100, b.mean() * 100

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.nboot, n))
    boot = (a[idx].mean(1) - b[idx].mean(1)) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    ah, bh = a >= 0.5, b >= 0.5
    bc = int((ah & ~bh).sum()); cc = int((~ah & bh).sum())

    print(f"paired HR@{args.k} on {n} samples ({la}: {na} run(s), {lb}: {nb} run(s))")
    print(f"  {la} {hr_a:.2f}   {lb} {hr_b:.2f}   delta {hr_a-hr_b:+.2f}   "
          f"95% CI [{lo:+.2f}, {hi:+.2f}]   p {p:.4f}")
    print(f"  {'significant' if p < 0.05 else 'not significant'} | "
          f"discordant {la}-only {bc}, {lb}-only {cc}")


if __name__ == "__main__":
    main()
