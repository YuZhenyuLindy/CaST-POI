"""Full-vocabulary ranking metrics.

Metrics are returned per sample, so the reported mean and any paired test are
computed from the same array.
"""
from typing import Dict, List, Sequence

import numpy as np

# K values written to metrics.json, independent of `eval_ks`, which only selects
# the validation metric.
REPORT_KS = (1, 5, 10, 20)
import torch


def per_sample_ranks(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """1-indexed rank of the target under full-vocabulary scoring.

    Ties use the mid-rank convention

        rank = 1 + #{strictly higher} + #{tied} / 2

    rather than 1 + #{strictly higher}. The optimistic rule is only equivalent
    when scores are dense: a count-based scorer assigns exactly 0 to every
    unvisited POI, so an unvisited target ties with thousands of items and would
    be credited with a near-top rank. Dense neural scorers are unaffected, so the
    optimistic rule would favour whichever method produces more ties.

    Non-finite scores raise instead of ranking, because every comparison with NaN
    is False and a diverged model would otherwise be scored as rank 1 everywhere,
    i.e. a perfect result.
    """
    if not torch.isfinite(scores).all():
        n_bad = int((~torch.isfinite(scores)).sum())
        raise ValueError(
            f"{n_bad} of {scores.numel()} scores are NaN or Inf. Ranking them would "
            f"report rank 1 for every sample (every comparison with NaN is False), "
            f"i.e. a perfect 100% score from a broken model. Check for a diverged "
            f"loss or an unsupported device dtype.")
    tgt = scores.gather(1, targets.unsqueeze(1))
    greater = (scores > tgt).sum(1)
    tied = (scores == tgt).sum(1) - 1          # exclude the target itself
    return greater + 1 + tied.float() / 2.0


def metrics_from_ranks(ranks: np.ndarray, ks: Sequence[int] = (5, 10, 20)) -> Dict[str, np.ndarray]:
    """Per-sample metric vectors. Mean of each vector is the reported number."""
    out: Dict[str, np.ndarray] = {}
    for k in ks:
        out[f"HR@{k}"] = (ranks <= k).astype(np.float64)
        out[f"NDCG@{k}"] = np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
    out["MRR"] = 1.0 / ranks
    return out


def summarize(per_sample: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {k: float(v.mean()) for k, v in per_sample.items()}


def format_metrics(m: Dict[str, float], pct: bool = True) -> str:
    order = sorted(m.keys(), key=lambda s: (s.split("@")[0], int(s.split("@")[1]) if "@" in s else 0))
    return " | ".join(f"{k}: {m[k] * 100:.2f}" if pct else f"{k}: {m[k]:.4f}" for k in order)


