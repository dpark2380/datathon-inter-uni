# Credit default prediction — inter-uni datathon

Predicting the probability that a customer defaults on their next payment,
scored on **binary log loss**. 24,000 labelled training customers, 6,000 in a
hidden test set.

**Final result: 0.40982** on the hidden test set, from `submissions/submission_tabpfn6.csv`.

## The model

A plain weighted average of three models' probabilities:

| Weight | Model | What it reads |
|---|---|---|
| 0.45 | LightGBM | 81 engineered features — the aggregate view of a customer |
| 0.30 | Bidirectional GRU | the six monthly statements in order — the trajectory |
| 0.25 | TabPFN | the same features, through a prior pretrained on synthetic data |

No calibration or post-processing beyond clipping to `[1e-4, 1-1e-4]`. Four
calibration methods were tested and all made the score worse; the probabilities
are already well calibrated (see `output/analysis/reliability_diagram.png`).

## Reproducing the submission

### Without retraining (seconds)

Saved prediction vectors are committed, so the exact submitted file can be
rebuilt without running any model. See `artifacts/tabpfn_6rep/README.md`
(verified: maximum absolute difference 9.71e-17 across all 6,000 rows).

The out-of-fold vectors are committed too, so the **blend weight search is also
reproducible in seconds** — it re-derives `(0.45, 0.30, 0.25)` at OOF log loss
0.421358 from `artifacts/oof_lgbm.npy`, `artifacts/oof_seq.npy` and
`artifacts/tabpfn_6rep/oof_tabpfn_6rep.npy` without retraining anything.

### From scratch

```bash
uv sync                                                      # installs everything in pyproject.toml
uv run python models/lgbm_model.py                           # LightGBM  ~10 min
uv run python models/seq_model.py                            # GRU       ~50 min
TABPFN_TOKEN="<token>" uv run python models/tabpfn_model.py  # TabPFN    ~1 h on Apple GPU
uv run python models/blend.py                                # weight search + writes the submission
```

`models/blend.py` writes `.output/predictions_blend.csv`. **That is the submitted
file** — it reproduces `submissions/submission_tabpfn6.csv` exactly (verified: maximum
absolute difference 0.0 across all 6,000 rows).

Order matters: `models/blend.py` reads the vectors the three model scripts save.

### TabPFN needs a token

TabPFN downloads pretrained weights and requires a Prior Labs licence and API
key from https://ux.priorlabs.ai — accept the licence on the **Licenses** tab,
then set `TABPFN_TOKEN`. Note that having a key is not enough on its own; the
licence acceptance is a separate step.

It runs on the Apple GPU (`mps`) by default, which is roughly **7× faster than
CPU** — 58 minutes against 3h 10m for identical predictions.

## Seeds and settings

Everything is deterministic given these:

| Setting | Value | Where |
|---|---|---|
| Fold seed | `SEED = 0` | `models/common.py` |
| Fold partitions | `REPEATS = range(6)` — 6 × 5-fold stratified | `models/common.py` |
| LightGBM seeds | `(0, 1, 2)` per fold → 90 models | `models/lgbm_model.py` |
| GRU seeds | `range(5)` per fold → 150 models | `models/seq_model.py` |
| TabPFN | 6 partitions → 30 fits, `n_estimators=4` | `models/tabpfn_model.py` |
| Blend weights | searched on OOF log loss, 0.05 grid | `models/blend.py` |

All three models additionally refit on all 24,000 rows and average that with
the fold ensemble for the test predictions.

## Files

**The pipeline, in execution order**

| # | File | Role |
|---|---|---|
| 1 | `1_EDA.ipynb` | Exploratory analysis — distributions, the target base rate, how each raw column behaves |
| 2 | `2_Column_Inspection.ipynb` | Column-level inspection (unique values, null counts) that identified the undocumented category codes |
| 3 | `models/common.py` | The spine: `clean()`, `features()` (81 features), `load()`, `folds()`, `score()`, `save()`. Every script below calls it |
| 4 | `models/lgbm_model.py` | LightGBM — 90 fold fits + 3-seed full-data refit |
| 5 | `models/seq_model.py` | Bidirectional GRU over the 6 × 8 monthly panel — 150 fits + refit |
| 6 | `models/tabpfn_model.py` | TabPFN — 30 fits + refit |
| 7 | `models/blend.py` | Weight search, calibration check, **writes the final submission** |

Steps 4–6 are independent of each other and can run in any order; step 7 reads
the `.npy` vectors they save. Steps 1–2 are documentation of how the cleaning
decisions were reached — the cleaning itself is `common.clean()`, so nothing
needs to be run before step 3.

**Analysis (not on the submission path)**

| File | Role |
|---|---|
| `models/interpret.py` | SHAP explanations and the reliability diagram |
| `models/fairness_audit.py` | Per-subgroup calibration and ranking, by sex, education, marriage and age |

**Tested and rejected** — `models/rejected/`, nine scripts kept as a record of
what was tried. Each was rejected by the same weight search that set the final
weights. See `models/rejected/README.md` for the table, and
`docs/experiment-ledger.html` for the measured deltas.

**Data and artifacts**

| Path | What it is |
|---|---|
| `datasets/train.csv`, `datasets/test.csv` | Competition-provided, unmodified |
| `datasets/train_clean.csv` | `train.csv` with the two category substitutions of `common.clean()`. Committed; regenerable in one command (see `REPORT.md` §4) |
| `artifacts/oof_lgbm.npy`, `artifacts/oof_seq.npy`, `artifacts/tabpfn_6rep/` | Out-of-fold vectors — enough to re-run the blend weight search in seconds without retraining |
| `artifacts/test_lgbm_full.npy`, `artifacts/test_seq.npy`, `artifacts/tabpfn_6rep/` | Test vectors — enough to rebuild the exact submitted file |
| `submissions/` | Every submitted prediction file; `submission_tabpfn6.csv` is the final one |
| `output/analysis/` | SHAP plots, reliability diagram, fairness audit |

**Documentation** — `REPORT.md` (full submission report), `Methodology.md`, and
`docs/` (published at
[dpark2380.github.io/datathon-inter-uni](https://dpark2380.github.io/datathon-inter-uni/)).

## Disclosure

- **Pretrained model:** TabPFN (Prior Labs), via the `tabpfn` package. Requires
  a licence and API token. It performs in-context learning and fits no
  parameters to this dataset.
- **AI coding agent:** Claude Code, used throughout.
- **External solutions consulted:** the 1st, 2nd and 3rd place write-ups from
  the AMEX Default Prediction competition. Five techniques were tested from them
  — DART boosting, high `min_data_in_leaf`, `feature_fraction_bynode`,
  recency-window aggregates, within-customer ranks. **None were adopted**; all
  measured worse under cross-validation.
- **External datasets:** none. Only the competition-provided files.
- **Manual modification of predictions:** none beyond the clipping noted above.

## Known limitations

- `EDUCATION="other"` (387 customers, 28 defaults) is over-predicted by ~61%
  relative and ranked poorly (AUC 0.645 against ~0.79 elsewhere). That category
  is where `clean()` collapses undocumented codes 0, 5 and 6, so it is a
  heterogeneous catch-all. It is the one subgroup where the model performs worse
  than predicting that group's own base rate.
- A meaningful share of defaults appear unpredictable from these columns: the 20
  worst-predicted customers are all defaulters with spotless payment histories.
- The blend weights are fitted to out-of-fold data, which is the one step in the
  pipeline where selection — and so overfitting risk — genuinely occurs. The
  weight surface is flat and shrinking toward uniform made the score worse, both
  of which suggest the fit is stable.
