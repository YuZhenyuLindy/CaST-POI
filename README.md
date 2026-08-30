# CaST-POI

Code for **"CaST-POI: Candidate-Conditioned Spatiotemporal Ranking for Next POI
Recommendation"** (ICDM 2026).

CaST-POI is a single-stage next-POI ranker. Each candidate acts as the query over
the user's trajectory through candidate-conditioned cross-attention with
candidate-relative temporal and spatial biases, on top of a causal self-attention
backbone and an explicit revisit gate. Every model reported in the paper, ours and
the baselines, is trained and evaluated under one pipeline: per-user leave-one-out
with full-vocabulary ranking.

## Layout

```
castpoi/              model and training/evaluation harness
  model.py            CaST-POI ranker
  layers.py           shared encoders, revisit features, loss
  config.py           configuration
  official.py         loading of the official preprocessed files
  data.py             datasets and dataloaders
  engine.py           training and full-vocabulary evaluation
  metrics.py          ranking metrics
  utils.py            seeding, device, IO
  timeparse.py        local-time handling
data/download.py      download the raw Foursquare / Gowalla check-ins
build_loo_split.py    build the per-user leave-one-out split
run.py                train / evaluate CaST-POI, and the ablation
heuristic.py          zero-parameter revisit heuristic
paired_sig.py         paired bootstrap significance test on HR@k
notebooks/            the Colab notebooks the reported results were produced with
DATA.md               how to obtain and verify the data
```

## Install

```
pip install -r requirements.txt
```

## Data

See [DATA.md](DATA.md). In short:

```
python data/download.py --dataset all          # raw sources, optional
python build_loo_split.py --src data_official --out data_loo --datasets nyc tky ca
```

`official.py` prints a content hash for each dataset at load time. A correct build
gives `73c596f2920e5266` (NYC), `f513cb5fcbe0832e` (TKY), `3b6a753dab84043c` (CA).
A different hash means the split is not the one the reported numbers use.

## Reproduce

```
# CaST-POI, three seeds, full-vocabulary ranking
OFFICIAL_DIR=data_loo python run.py --dataset all --seeds 42 43 44

# component ablation
OFFICIAL_DIR=data_loo python run.py --dataset all --ablation

# zero-parameter revisit heuristic
OFFICIAL_DIR=data_loo python heuristic.py --dataset all
```

Metrics and per-sample ranks are written under `results/`.

### Baselines

The seven sequential baselines in Table 1 (SASRec, BERT4Rec, NARM, STAMP, Caser,
SR-GNN, CORE) are trained with the official RecBole implementations on the same
`data_loo` split. That pipeline lives in
[`notebooks/recbole_baselines_loo.ipynb`](notebooks/recbole_baselines_loo.ipynb),
which writes `run_recbole.py` and runs it for every model, dataset and seed. It
exports our exact test samples to RecBole, scores them full-vocabulary, applies
the same mid-rank tie convention, and reorders the ranks into our test order via
the official check-in id, so the outputs are directly comparable to `run.py`'s.

### Significance

```
python paired_sig.py --a results/runs/castpoi_loo_lr0.001/nyc/full \
                     --b <baseline_run_dir> --k 10
```

Both run directories must contain `check_ins_id.npy` alongside `ranks.npy`; the
test aligns the two rank vectors on that id rather than on position.

## Notebooks

`notebooks/` holds the Colab notebooks the reported numbers were produced with:
`castpoi_loo_seed42.ipynb`, `castpoi_loo_seed43.ipynb` and
`castpoi_loo_seed44.ipynb` for CaST-POI and the heuristic, and
`recbole_baselines_loo.ipynb` for the baselines. They are self-contained: each
writes out the `castpoi/` package and the runners before executing them, and the
package they write is byte-identical to the one in this repository.

## Citation

```bibtex
@inproceedings{castpoi2026,
  title     = {CaST-POI: Candidate-Conditioned Spatiotemporal Ranking for Next POI Recommendation},
  booktitle = {IEEE International Conference on Data Mining (ICDM)},
  year      = {2026}
}
```
