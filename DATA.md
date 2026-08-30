# Data

Three benchmarks are used: NYC and TKY from Foursquare TSMC2014 (Yang et al.,
2015) and CA from Gowalla (Cho et al., 2011).

## 1. Preprocessed files (`data_official/`)

We do not run our own filtering. The starting point is the preprocessed output of
the STHGCN / LLM4POI pipeline, which already applies:

| step | value |
|---|---|
| `min_poi_freq` / `min_user_freq` | 9 / 9, with `count > freq` semantics (keep >= 10) |
| CA | the filter is applied twice |
| NYC | not filtered here; it arrives pre-split from GETNext |
| split | global chronological 80/10/10 over all check-ins |
| trajectories | 24-hour session gap, singleton sessions dropped |
| unseen entities | val/test rows whose user or POI is absent from train are removed |

Obtain them either from the published `w11wo/LLM4POI` release or by running
LLM4POI's own `preprocess()` unmodified. Place the result as

```
data_official/<nyc|tky|ca>/train_sample.csv
data_official/<nyc|tky|ca>/validate_sample_with_traj.csv
data_official/<nyc|tky|ca>/test_sample_with_traj.csv
```

`data/download.py` fetches the raw source archives if you prefer to start there.

## 2. Leave-one-out split (`data_loo/`)

All results in the paper use a per-user leave-one-out split built from the files
above:

```
python build_loo_split.py --src data_official --out data_loo --datasets nyc tky ca
```

Per user, ordered by time, the last check-in is test, the second-to-last is
validation, and the rest are train. Users with fewer than three check-ins are
dropped, and val/test rows whose POI is unseen in the new train split are removed.

## 3. Verifying the split

`official.py` prints a content hash at load time. A correct build gives

| dataset | fingerprint | users | POIs | test instances |
|---|---|---|---|---|
| NYC | `73c596f2920e5266` | 1,047 | 4,974 | 1,044 |
| TKY | `f513cb5fcbe0832e` | 2,280 | 7,832 | 2,280 |
| CA  | `3b6a753dab84043c` | 3,956 | 9,689 | 3,956 |

If a fingerprint differs, the split is not the one the reported numbers were
computed on and the results are not comparable.

## Notes on the source files

- `UTCTimeOffsetEpoch` is unusable in the published files; `timeparse.py` parses
  `UTCTimeOffset` instead. See that module for details.
- POI ids are label-encoded under two conventions: id 0 is a real POI in NYC and
  the padding bucket in TKY/CA. `official.py` handles both explicitly.
