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
rebuilt without running any model. See `artifacts/tabpfn_6rep/README.md`.

### From scratch

```bash
uv sync                          # installs everything in pyproject.toml
uv run python models/model.py           # LightGBM        ~10 min
uv run python models/seq_model.py       # GRU             ~50 min
TABPFN_TOKEN="<token>" uv run python models/tabpfn_model.py   # TabPFN  ~1 h on Apple GPU
uv run python models/blend.py           # weight search + writes the submission
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
| LightGBM seeds | `(0, 1, 2)` per fold → 90 models | `models/model.py` |
| GRU seeds | `range(5)` per fold → 150 models | `models/seq_model.py` |
| TabPFN | 6 partitions → 30 fits, `n_estimators=4` | `models/tabpfn_model.py` |
| Blend weights | searched on OOF log loss, 0.05 grid | `models/blend.py` |

All three models additionally refit on all 24,000 rows and average that with
the fold ensemble for the test predictions.

## Files

**The pipeline**

| File | Role |
|---|---|
| `models/common.py` | data loading, cleaning, 81-feature engineering, the shared CV splits |
| `models/model.py` | LightGBM |
| `models/seq_model.py` | bidirectional GRU over the 6-month panel |
| `models/tabpfn_model.py` | TabPFN |
| `models/blend.py` | weight search, calibration check, **writes the final submission** |

**Analysis**

| File | Role |
|---|---|
| `models/interpret.py` | SHAP explanations and the reliability diagram |
| `models/fairness_audit.py` | per-subgroup calibration and ranking, by sex, education, marriage and age |

**Tested and rejected** — kept as a record of what was tried, not part of the
pipeline. See `docs/experiment-ledger.html` for measured results.

`models/nn_model.py` (MLP) · `models/cnn_model.py` · `models/attn_model.py` (transformer, never
completed a full run) · `models/autoenc_model.py` · `models/multitask_model.py` ·
`models/survival_model.py` · `models/seq_spend_model.py` · `models/pseudo_label.py`

**Documentation** — `docs/`, and `artifacts/` for saved prediction vectors.

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
