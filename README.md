# CaST-POI

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg">
  <img alt="Venue" src="https://img.shields.io/badge/ICDM-2026-6a5acd.svg">
</p>

Official implementation of **"CaST-POI: Candidate-Conditioned Spatiotemporal Ranking for Next POI Recommendation"** (ICDM 2026).

CaST-POI is a single-stage next-POI ranker. Each candidate acts as the query over the user's trajectory through candidate-conditioned cross-attention with candidate-relative temporal and spatial biases, on top of a causal self-attention backbone and an explicit revisit gate. Every model reported in the paper — ours and the baselines — is trained and evaluated under one pipeline: **per-user leave-one-out with full-vocabulary ranking**.

---

## Installation

```bash
git clone https://github.com/<user>/castpoi.git
cd castpoi
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch 2.0+. A single GPU is enough; the largest model is 1.7M parameters.

## Data

Full instructions in [DATA.md](DATA.md). The starting point is the preprocessed output of the STHGCN / LLM4POI pipeline, placed under `data_official/<ds>/`. Once it is in place:

```bash
python build_loo_split.py --src data_official --out data_loo --datasets nyc tky ca
```

`data/download.py` fetches the raw Foursquare / Gowalla archives if you would rather start from source, but LLM4POI's own preprocessing still has to be run on them before `build_loo_split.py` can be used.

`official.py` prints a content hash for each dataset at load time. Check it before trusting any number:

| Dataset | Fingerprint | Users | POIs | Test instances |
|:--|:--|--:|--:|--:|
| NYC | `73c596f2920e5266` | 1,047 | 4,974 | 1,044 |
| TKY | `f513cb5fcbe0832e` | 2,280 | 7,832 | 2,280 |
| CA  | `3b6a753dab84043c` | 3,956 | 9,689 | 3,956 |

A different hash means the split is not the one the reported numbers were computed on.

## Reproduce

```bash
# CaST-POI, three seeds, full-vocabulary ranking
OFFICIAL_DIR=data_loo python run.py --dataset all --seeds 42 43 44

# component ablation (full / no_repeat / no_candcond / no_spatial_bias / no_temporal_bias / no_backbone)
OFFICIAL_DIR=data_loo python run.py --dataset all --ablation

# zero-parameter revisit heuristic
OFFICIAL_DIR=data_loo python heuristic.py --dataset all
```

Metrics and per-sample ranks are written under `results/`.

<details>
<summary><b>Baselines</b></summary>

The seven sequential baselines are trained with the official RecBole implementations on the same `data_loo` split. That pipeline is in [`notebooks/recbole_baselines_loo.ipynb`](notebooks/recbole_baselines_loo.ipynb), which writes `run_recbole.py` and runs it for every model, dataset and seed. It exports our exact test samples to RecBole, scores them full-vocabulary, applies the same mid-rank tie convention, and reorders the ranks into our test order via the official check-in id — so the outputs are directly comparable to `run.py`'s.

</details>

<details>
<summary><b>Significance testing</b></summary>

```bash
python paired_sig.py --a results/runs/castpoi_loo_lr0.001/nyc/full \
                     --b <baseline_run_dir> --k 10
```

Both run directories must contain `check_ins_id.npy` next to `ranks.npy`. The test aligns the two rank vectors on that id rather than on position, so it is valid across independently produced runs.

</details>

## Results

Per-user leave-one-out with full-vocabulary ranking, mean over seeds 42/43/44 (%). Best per column in **bold**. *Improv.* is the relative gain of CaST-POI over the strongest baseline in that column, with a paired Wilcoxon test Holm-corrected across the fifteen headline comparisons (<sup>&ast;</sup> *p* < 0.05, <sup>&ast;&ast;</sup> *p* < 0.01, <sup>&ast;&ast;&ast;</sup> *p* < 0.001; no marker = not significant).

<b>NYC</b>

| Method | HR@5 | HR@10 | NDCG@5 | NDCG@10 | MRR |
|:--|--:|--:|--:|--:|--:|
| SASRec | 50.42 | 56.99 | 38.01 | 40.14 | 35.35 |
| BERT4Rec | 47.67 | 54.69 | 36.14 | 38.43 | 33.78 |
| Caser | 48.12 | 54.15 | 35.84 | 37.80 | 33.12 |
| NARM | 49.39 | 55.87 | 37.15 | 39.30 | 34.55 |
| SR-GNN | 44.06 | 52.81 | 32.82 | 35.67 | 30.90 |
| STAMP | 41.60 | 46.33 | 32.44 | 33.98 | 30.51 |
| CORE | **51.31** | 57.76 | 38.30 | 40.40 | 35.38 |
| Revisit heuristic | 41.19 | 54.60 | 28.66 | 33.05 | 26.94 |
| **CaST-POI** | 51.02 | **58.01** | **39.51** | **41.80** | **37.20** |
| *Improv.* | −0.57% | +0.43% | +3.16% | +3.47% | +5.14% |

<b>TKY</b>

| Method | HR@5 | HR@10 | NDCG@5 | NDCG@10 | MRR |
|:--|--:|--:|--:|--:|--:|
| SASRec | 46.61 | 56.02 | 34.09 | 37.14 | 32.12 |
| BERT4Rec | 44.28 | 53.67 | 33.40 | 36.46 | 31.96 |
| Caser | 43.60 | 51.81 | 33.18 | 35.86 | 31.69 |
| NARM | 45.04 | 53.55 | 34.15 | 36.92 | 32.57 |
| SR-GNN | 45.44 | 53.49 | 34.15 | 36.74 | 32.33 |
| STAMP | 42.87 | 49.82 | 32.91 | 35.17 | 31.40 |
| CORE | 43.41 | 53.48 | 29.77 | 33.05 | 27.58 |
| Revisit heuristic | 29.25 | 41.54 | 19.29 | 23.22 | 18.97 |
| **CaST-POI** | **48.39** | **57.76** | **36.73** | **39.76** | **35.05** |
| *Improv.* | +3.82%<sup>&ast;</sup> | +3.11%<sup>&ast;</sup> | +7.55%<sup>&ast;&ast;&ast;</sup> | +7.05%<sup>&ast;&ast;&ast;</sup> | +7.61%<sup>&ast;&ast;&ast;</sup> |

<b>CA</b>

| Method | HR@5 | HR@10 | NDCG@5 | NDCG@10 | MRR |
|:--|--:|--:|--:|--:|--:|
| SASRec | 35.48 | 44.25 | 24.81 | 27.65 | 23.69 |
| BERT4Rec | 28.99 | 36.80 | 21.64 | 24.18 | 21.37 |
| Caser | 27.32 | 34.34 | 20.23 | 22.51 | 19.88 |
| NARM | 32.01 | 40.09 | 23.66 | 26.28 | 23.11 |
| SR-GNN | 30.74 | 38.29 | 22.75 | 25.19 | 22.22 |
| STAMP | 27.92 | 34.50 | 21.11 | 23.24 | 20.80 |
| CORE | 31.66 | 39.92 | 23.10 | 25.77 | 22.57 |
| Revisit heuristic | 28.64 | 35.52 | 21.13 | 23.35 | 20.05 |
| **CaST-POI** | **38.17** | **46.29** | **28.23** | **30.86** | **27.12** |
| *Improv.* | +7.58%<sup>&ast;&ast;&ast;</sup> | +4.61%<sup>&ast;&ast;</sup> | +13.78%<sup>&ast;&ast;&ast;</sup> | +11.61%<sup>&ast;&ast;&ast;</sup> | +14.48%<sup>&ast;&ast;&ast;</sup> |

The gains are significant on TKY and CA. On NYC they are not: CaST-POI and CORE are statistically indistinguishable, and CORE is ahead on HR@5. The paper reports this rather than restricting the comparison to the datasets where the test passes.

## Repository layout

```
.
├── castpoi/                         # model and training/evaluation harness
│   ├── __init__.py
│   ├── model.py                     # CaST-POI ranker
│   ├── layers.py                    # shared encoders, revisit features, loss
│   ├── config.py                    # configuration
│   ├── official.py                  # loading of the official preprocessed files
│   ├── data.py                      # datasets and dataloaders
│   ├── engine.py                    # training and full-vocabulary evaluation
│   ├── metrics.py                   # ranking metrics
│   ├── timeparse.py                 # local-time handling
│   └── utils.py                     # seeding, device, IO
├── data/
│   └── download.py                  # download the raw Foursquare / Gowalla check-ins
├── notebooks/                       # the notebooks the reported results were produced with
│   ├── castpoi_loo_seed42.ipynb
│   ├── castpoi_loo_seed43.ipynb
│   ├── castpoi_loo_seed44.ipynb
│   └── recbole_baselines_loo.ipynb
├── build_loo_split.py               # build the per-user leave-one-out split
├── run.py                           # train / evaluate CaST-POI, and the ablation
├── heuristic.py                     # zero-parameter revisit heuristic
├── paired_sig.py                    # paired bootstrap significance test on HR@k
├── requirements.txt
└── DATA.md                          # how to obtain and verify the data
```

## Notebooks

`notebooks/` holds the Colab notebooks the reported numbers were produced with: `castpoi_loo_seed42.ipynb`, `castpoi_loo_seed43.ipynb` and `castpoi_loo_seed44.ipynb` for CaST-POI and the heuristic, and `recbole_baselines_loo.ipynb` for the baselines.

They are self-contained: each writes out the `castpoi/` package and the runners before executing them. The package they write is byte-identical to the one in this repository.

## Citation

```bibtex
@inproceedings{yu2026castpoi,
  title     = {CaST-POI: Candidate-Conditioned Spatiotemporal Ranking for Next POI Recommendation},
  author    = {Yu, Zhenyu and Meng, Chunlei and Zeng, Yangchen and
               Idris, Mohd Yamani Idna and Guan, Jihong and Zhou, Shuigeng},
  booktitle = {IEEE International Conference on Data Mining (ICDM)},
  year      = {2026}
}
```
