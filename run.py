"""Train and evaluate CaST-POI under per-user leave-one-out + full-vocabulary ranking.

    python build_loo_split.py --src data_official --out data_loo
    OFFICIAL_DIR=data_loo python run.py --dataset all --seeds 42 43 44
    OFFICIAL_DIR=data_loo python run.py --dataset all --ablation
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("CASTPOI_RESULTS", HERE / "results"))
OFFICIAL_DIR = Path(os.environ.get("OFFICIAL_DIR", HERE / "data_loo"))

from castpoi.config import resolve_config
from castpoi.official import load_official
from castpoi.data import create_dataloaders
from castpoi.engine import train_model, evaluate, measure_inference
from castpoi.metrics import format_metrics
from castpoi.utils import set_seed, pick_device, count_params
from castpoi.model import CaSTPOI

DATASETS = ("nyc", "tky", "ca")
SEEDS = [42, 43, 44]

ABLATION_VARIANTS = {
    "full":             {},
    "no_repeat":        {"use_repeat": False},
    "no_candcond":      {"castpoi_candidate_conditioned": False},
    "no_spatial_bias":  {"castpoi_spatial_bias": False},
    "no_temporal_bias": {"castpoi_temporal_bias": False},
    "no_backbone":      {"use_backbone": False},
}


def build_config(dataset, quick, lr):
    cfg = resolve_config(dataset, {})
    cfg["eval_ks"] = [1, 5, 10, 20]
    cfg["eval_batch_size"] = 256
    cfg["castpoi_cand_chunk"] = 1024
    cfg["train_objective"] = "full"
    cfg["batch_size"] = min(cfg["batch_size"], 128)
    if lr is not None:
        cfg["learning_rate"] = lr
    if quick:
        cfg["num_epochs"] = 2
    return cfg


def run_one(dataset, data, cfg, seed, device, tag, variant="full"):
    out_dir = RESULTS / "runs" / tag / dataset / variant / f"seed{seed}"
    if (out_dir / "metrics.json").exists():
        print(f"[skip] {dataset}/{variant}/seed{seed}")
        return json.load(open(out_dir / "metrics.json"))["test"]
    set_seed(seed)
    model = CaSTPOI(data["num_pois"], data["poi_locations"], cfg)
    tl, vl, tel = create_dataloaders(data, cfg)
    t0 = time.time()
    model, hist = train_model(model, tl, vl, cfg, device)
    test_metrics, _, extras = evaluate(model, tel, device, cfg["eval_ks"], collect_topk=True)
    wall = time.time() - t0

    ranks = extras.get("ranks"); rep_expl = None
    if ranks is not None:
        is_rep = []
        for b in tel:
            src = b.get("repeat_hist", b["poi_ids"])
            is_rep.append((src == b["target_poi"].unsqueeze(1)).any(1).cpu().numpy())
        is_rep = np.concatenate(is_rep).astype(bool)
        if len(is_rep) == len(ranks):
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "ranks.npy", ranks); np.save(out_dir / "is_repeat.npy", is_rep)
            if extras.get("check_ins_id") is not None:
                np.save(out_dir / "check_ins_id.npy", extras["check_ins_id"])

            def sub(m):
                r = ranks[m]
                if not len(r): return {"n": 0}
                return {**{f"HR@{k}": float((r <= k).mean()) for k in cfg["eval_ks"]},
                        "MRR": float((1 / r).mean()), "n": int(len(r))}
            rep_expl = {"repeat": sub(is_rep), "explore": sub(~is_rep),
                        "repeat_frac": float(is_rep.mean())}
    try:
        eff = measure_inference(model, tel, device, cfg["eval_batch_size"])
    except Exception as e:
        eff = {"error": str(e)}
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"dataset": dataset, "model": "castpoi", "variant": variant, "split": "loo", "seed": seed,
           "num_pois": data["num_pois"], "data_fingerprint": data["stats"].get("data_fingerprint"),
           "params": count_params(model), "test": test_metrics, "repeat_explore": rep_expl,
           "efficiency": eff, "best_val_hr10": hist.get("best_val_hr10"), "wall_seconds": wall,
           "config": {k: cfg[k] for k in ("learning_rate", "batch_size", "num_epochs",
                                          "train_objective", "eval_ks")}}
    json.dump(rec, open(out_dir / "metrics.json", "w"), indent=2)
    print(f"[{dataset}/{variant}/seed{seed}] {format_metrics(test_metrics)} | {wall/60:.1f}min")
    return test_metrics


def run_dataset(dataset, seeds, device, quick, lr):
    tag = "castpoi_loo" + ("" if lr is None else f"_lr{lr:g}")
    data = load_official(dataset, OFFICIAL_DIR)
    cfg = build_config(dataset, quick, lr)
    ms = [run_one(dataset, data, cfg, s, device, tag) for s in seeds]
    agg = {k: {"mean": float(np.mean([m[k] for m in ms])), "std": float(np.std([m[k] for m in ms]))}
           for k in ms[0]}
    print("  mean: " + " ".join(f"{k} {v['mean']*100:.2f}" for k, v in agg.items()))
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump({"dataset": dataset, "split": "loo", "seeds": seeds,
               "data_fingerprint": data["stats"].get("data_fingerprint"),
               "results": {"full": agg}}, open(RESULTS / f"castpoi_loo_{dataset}.json", "w"), indent=2)


def run_ablation(dataset, seeds, device, quick, lr, variants):
    tag = "castpoi_loo_ablation" + ("" if lr is None else f"_lr{lr:g}")
    data = load_official(dataset, OFFICIAL_DIR)
    base = build_config(dataset, quick, lr)
    summary = {}
    for v in variants:
        cfg = dict(base); cfg.update(ABLATION_VARIANTS[v])
        ms = [run_one(dataset, data, cfg, s, device, tag, variant=v) for s in seeds]
        summary[v] = {k: {"mean": float(np.mean([m[k] for m in ms])),
                          "std": float(np.std([m[k] for m in ms]))} for k in ms[0]}
    if "full" in summary:
        base_hr = summary["full"]["HR@10"]["mean"]
        for v in variants:
            d = (summary[v]["HR@10"]["mean"] - base_hr) * 100
            print(f"  {v:18} HR@10 {summary[v]['HR@10']['mean']*100:6.2f}  d{d:+.2f}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump({"dataset": dataset, "split": "loo", "kind": "ablation", "seeds": seeds,
               "data_fingerprint": data["stats"].get("data_fingerprint"), "results": summary},
              open(RESULTS / f"castpoi_loo_ablation_{dataset}.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--variants", nargs="+", default=list(ABLATION_VARIANTS))
    args = ap.parse_args()
    for v in args.variants:
        assert v in ABLATION_VARIANTS, f"unknown variant {v}"
    assert OFFICIAL_DIR.exists(), f"OFFICIAL_DIR={OFFICIAL_DIR} not found; build data_loo first."
    device = pick_device(args.device)
    seeds = [42] if args.quick else args.seeds
    for ds in (list(DATASETS) if args.dataset == "all" else [args.dataset]):
        if args.ablation:
            run_ablation(ds, seeds, device, args.quick, args.lr, args.variants)
        else:
            run_dataset(ds, seeds, device, args.quick, args.lr)


if __name__ == "__main__":
    main()
