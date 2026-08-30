"""Model and training configuration.

One BASE_CONFIG is used for every dataset; there are no per-dataset defaults.
Preprocessing and splitting are not configurable here. They come from the
official files loaded by official.py and from build_loo_split.py; see DATA.md.
"""
import copy
from typing import Any, Dict

BASE_CONFIG: Dict[str, Any] = {
    # Model
    "poi_embed_dim": 128,
    "slot_embed_dim": 16,
    "spatial_dim": 32,
    "dist_embed_dim": 16,
    "num_dist_buckets": 8,
    # Revisit gate: MLP width, and the window the visit counts are taken over.
    # The sequence encoder still sees only max_history_len; counting is cheap,
    # so this window can be longer.
    "repeat_gate_hidden": 32,
    "repeat_history_len": 512,

    # Training objective. "sampled" scores the positive against `num_negatives`
    # sampled negatives; "full" scores it against the entire POI vocabulary.
    # Evaluation is always full-vocabulary, and the RecBole baselines train with
    # full-vocabulary cross-entropy, so run.py sets "full". The vocabulary is
    # small enough (4,980 / 7,832 / 9,689) for a dense softmax.
    "train_objective": "sampled",   # "sampled" | "full"
    "num_negatives": 499,           # ignored when train_objective == "full"
    "batch_size": 512,
    "eval_batch_size": 1024,
    "num_epochs": 50,
    "learning_rate": 2e-3,
    "weight_decay": 1e-4,
    "dropout": 0.1,
    "gradient_clip": 5.0,
    "early_stopping_patience": 10,
    "warmup_epochs": 3,
    "label_smoothing": 0.02,
    "explore_weight": 1.5,
    "bpr_weight": 0.5,
    "bpr_margin": 1.0,

    # Model input truncation, not a data filter.
    "max_history_len": 50,
    # Selects the validation metric. run.py widens this to [1, 5, 10, 20] so the
    # saved metrics cover every K the tables use.
    "eval_ks": [5, 10],
}

DATASETS = ("nyc", "tky", "ca")


def resolve_config(dataset: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Effective config for `dataset`. Identical for every dataset by default."""
    ds = dataset.lower()
    if ds not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {list(DATASETS)}")
    cfg = copy.deepcopy(BASE_CONFIG)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["dataset"] = ds

    for gone in ("split_protocol", "test_size", "min_poi_checkins", "min_user_checkins"):
        if gone in cfg:
            raise ValueError(
                f"{gone!r} is not a configuration option: preprocessing comes from "
                f"the official STHGCN/LLM4POI output, not from this package. "
                f"See DATA.md.")
    return cfg
