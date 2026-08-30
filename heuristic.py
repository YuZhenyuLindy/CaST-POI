"""Zero-parameter revisit heuristic under LOO + full-vocabulary ranking.

Scores each candidate by (visit count in the user's history) x (global popularity),
so only previously visited POIs can be ranked. Mid-rank tie convention, matching the
learned-model evaluation.

    OFFICIAL_DIR=data_loo python heuristic.py --dataset all
"""
import argparse, json, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("CASTPOI_RESULTS", HERE / "results"))
OFFICIAL_DIR = Path(os.environ.get("OFFICIAL_DIR", HERE / "data_loo"))

from castpoi.config import resolve_config
from castpoi.official import load_official
from castpoi.data import create_dataloaders

DATASETS = ("nyc", "tky", "ca")
EVAL_KS = [1, 5, 10, 20]


def run(dataset):
    cfg = resolve_config(dataset, {})
    cfg["eval_batch_size"] = 256
    data = load_official(dataset, OFFICIAL_DIR)
    num_pois = data["num_pois"]
    _, _, tel = create_dataloaders(data, cfg)

    pop = np.zeros(num_pois, dtype=np.float64)
    for b in tel:
        for row in b.get("repeat_hist", b["poi_ids"]).numpy():
            np.add.at(pop, row, 1.0)
    pop[0] = 0.0
    pop = pop / max(pop.sum(), 1.0)

    cid_all = tel.dataset.check_ins_ids
    cid_all = cid_all() if callable(cid_all) else cid_all
    ranks, is_rep, cids, ptr = [], [], [], 0
    for b in tel:
        src = b.get("repeat_hist", b["poi_ids"]).numpy()
        tgt = b["target_poi"].numpy()
        for i in range(src.shape[0]):
            cnt = np.zeros(num_pois, dtype=np.float64)
            np.add.at(cnt, src[i], 1.0)
            cnt[0] = 0.0
            score = cnt * pop
            t = int(tgt[i]); st = score[t]
            greater = int((score[1:] > st).sum())
            equal = int((score[1:] == st).sum())
            ranks.append(greater + (equal + 1) / 2.0)
            is_rep.append(cnt[t] > 0)
            cids.append(int(cid_all[ptr + i]))
        ptr += src.shape[0]

    ranks = np.array(ranks); is_rep = np.array(is_rep, dtype=bool); cids = np.array(cids, dtype=np.int64)

    def sub(mask):
        r = ranks[mask]
        if not len(r): return {"n": 0}
        return {**{f"HR@{k}": float((r <= k).mean()) for k in EVAL_KS},
                "MRR": float((1.0 / r).mean()), "n": int(len(r))}

    metrics = {**{f"HR@{k}": float((ranks <= k).mean()) for k in EVAL_KS},
               "MRR": float((1.0 / ranks).mean()), "n": int(len(ranks))}
    rep_expl = {"repeat": sub(is_rep), "explore": sub(~is_rep), "repeat_frac": float(is_rep.mean())}

    out = RESULTS / "runs" / "heuristic_loo" / dataset
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ranks.npy", ranks); np.save(out / "check_ins_id.npy", cids)
    np.save(out / "is_repeat.npy", is_rep)
    json.dump({"dataset": dataset, "model": "revisit_heuristic", "split": "loo",
               "data_fingerprint": data["stats"].get("data_fingerprint"),
               "test": metrics, "repeat_explore": rep_expl}, open(out / "metrics.json", "w"), indent=2)
    print(f"[{dataset}] " + " ".join(f"{k} {metrics[k]*100:.2f}"
          for k in ("HR@1", "HR@5", "HR@10", "HR@20", "MRR")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    args = ap.parse_args()
    assert OFFICIAL_DIR.exists(), f"OFFICIAL_DIR={OFFICIAL_DIR} not found; build data_loo first."
    for ds in (list(DATASETS) if args.dataset == "all" else [args.dataset]):
        run(ds)


if __name__ == "__main__":
    main()
