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

Full instructions in [DATA.md](DATA.md).

```bash
python data/download.py --dataset all                    # raw sources (optional)
python build_loo_split.py --src data_official --out data_loo --datasets nyc tky ca
```

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

HR@10 and MRR (%), per-user leave-one-out with full-vocabulary ranking, mean over seeds 42/43/44. Best per column in **bold**.

| Method | NYC HR@10 | NYC MRR | TKY HR@10 | TKY MRR | CA HR@10 | CA MRR |
|:--|--:|--:|--:|--:|--:|--:|
| SASRec | 56.99 | 35.35 | 56.02 | 32.12 | 44.25 | 23.69 |
| BERT4Rec | 54.69 | 33.78 | 53.67 | 31.96 | 36.80 | 21.37 |
| Caser | 54.15 | 33.12 | 51.81 | 31.69 | 34.34 | 19.88 |
| NARM | 55.87 | 34.55 | 53.55 | 32.57 | 40.09 | 23.11 |
| SR-GNN | 52.81 | 30.90 | 53.49 | 32.33 | 38.29 | 22.22 |
| STAMP | 46.33 | 30.51 | 49.82 | 31.40 | 34.50 | 20.80 |
| CORE | 57.76 | 35.38 | 53.48 | 27.58 | 39.92 | 22.57 |
| Revisit heuristic | 54.60 | 26.94 | 41.54 | 18.97 | 35.52 | 20.05 |
| **CaST-POI** | **58.01** | **37.20** | **57.76** | **35.05** | **46.29** | **27.12** |

Under a paired Wilcoxon test with Holm correction over the fifteen headline comparisons, the gains are significant on TKY and CA. On NYC they are not: CaST-POI and CORE are statistically indistinguishable on HR@*k*. The paper reports this rather than restricting the comparison to the datasets where the test passes.

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
