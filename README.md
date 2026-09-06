# Credit default prediction - Inter-uni datathon

**Final result: 0.40982** on the hidden test set, from `submissions/submission_tabpfn6.csv`.

## Our Datasets
In our datasets folder, we have the original given ```test.csv``` and ```train.csv``` files. We proceeded to clean the ```train.csv``` file, outputting the ```train_clean.csv``` file which was used for all of our models. 

## The model

A plain weighted average of three models' probabilities:

| Weight | Model | What it reads |
|---|---|---|
| 0.45 | LightGBM | Trained off 81 engineered features |
| 0.30 | Bidirectional GRU | Reads the 6 monthly repayments statements in order to understand the trend|
| 0.25 | TabPFN | Reads the same 81 engineered features as LightGBM but trained off synthetic priors|

These are constructed in the following files:
1. ```common.py```  
This file contains the functions which are called in every model. This includes ```clean()``` which cleans the dataset to the same standard, ```features()``` which calculates the features for each dataset, ```load()``` which loads the datasets and ```folds()``` which standardises the OOF predictions.
2. ```lgbm_model.py```  
This is the file which contains the LightGBM model
3. ```seg_model.py```  
This is the file which contains the bidirectional GRU model
4. ```tabpfn_model.py```  
This is the file which contains the TabPFN model
5. ```blend.py```  
This file reads the three vectors produced by the three models and uses a weight search grid to find the most optimal weights for the ensemble. 

## Reproducing the submission

### Without retraining

To save time, we have saved the prediction vectors such that the exact submitted file can be rebuilt without having to run any model. This was necessary as training our models took over 2 hours in total.The following files are provided  

| File | What it is |
|---|---|
| `tabpfn_model_6rep.py` | The exact script that generated the vectors below |
| `oof_tabpfn_6rep.npy` | Out-of-fold predictions, 24,000 rows (OOF log loss 0.42345) |
| `test_tabpfn_6rep.npy` | Test predictions, 6,000 rows (mean 0.2154) |

The following code can be run to reproduce the results without retraining.
```python
import numpy as np, pandas as pd
lgbm = np.load("artifacts/test_lgbm_full.npy")   # pure full-data refit, NOT test_lgbm.npy
seq  = np.load("artifacts/test_seq.npy")
tab  = np.load("artifacts/tabpfn_6rep/test_tabpfn_6rep.npy")
preds = np.clip(0.45*lgbm + 0.30*seq + 0.25*tab, 1e-4, 1-1e-4)
ids = pd.read_csv("datasets/test.csv", usecols=["client_id"], dtype={"client_id": str})["client_id"]
pd.DataFrame({"client_id": ids, "prob_default": preds}).to_csv("submissions/submission_tabpfn6.csv", index=False)
```

### Retraining from scratch

This requires a Prior Labs licence and API token (`TABPFN_TOKEN`) since TabPFN
downloads pretrained weights:

```bash
# 1. Install dependencies
uv sync                                                             # everything in pyproject.toml

# 2. Train the three models (2 and 3 can run in either order; TabPFN needs a token)
uv run python models/lgbm_model.py                                  # LightGBM        ~10 min
uv run python models/seq_model.py                                   # Bidirectional GRU  ~50 min

# TabPFN requires a Prior Labs licence + API token from https://ux.priorlabs.ai
# (accept the licence on the "Licenses" tab — the key alone is not enough)
TABPFN_TOKEN="<token>" uv run python models/tabpfn_model.py         # TabPFN  ~1h on Apple GPU (mps) / ~3h 10m on CPU

# 4. Blend — this writes the final submission
uv run python models/blend.py

```





The out-of-fold vectors are committed too, so the blend weight search is also
reproducible in seconds. It re-derives `(0.45, 0.30, 0.25)` at OOF log loss
0.421358 from `artifacts/oof_lgbm.npy`, `artifacts/oof_seq.npy` and
`artifacts/tabpfn_6rep/oof_tabpfn_6rep.npy` without retraining anything.

### TabPFN needs a token

Note that TabPFN downloads pretrained weights and requires a Prior Labs licence and API
key from https://ux.priorlabs.ai. Accept the licence on the **Licenses** tab,
then set `TABPFN_TOKEN`. 

## Seeds and settings



| Setting | Value | Where |
|---|---|---|
| Fold seed | `SEED = 0` | `models/common.py` |
| Fold partitions | `REPEATS = range(6)` -  6 × 5-fold stratified | `models/common.py` |
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
| 1 | `1_EDA.ipynb` | Exploratory analysis including distributions, the target base rate, how each raw column behaves |
| 2 | `2_Column_Inspection.ipynb` | Column level inspection (unique values, null counts) that identified the undocumented category codes |
| 3 | `models/common.py` | Functions such as `clean()`, `features()` (81 features), `load()`, `folds()`, `score()`, `save()`. Every script below calls it |
| 4 | `models/lgbm_model.py` | LightGBM - 90 fold fits + 3 seed full data refit |
| 5 | `models/seq_model.py` | Bidirectional GRU over the 6 × 8 monthly panel - 150 fits + refit |
| 6 | `models/tabpfn_model.py` | TabPFN - 30 fits + refit |
| 7 | `models/blend.py` | Weight search, calibration check, **writes the final submission** |

Steps 4–6 are independent of each other and can run in any order; step 7 reads
the `.npy` vectors they save. Steps 1–2 are documentation of how the cleaning
decisions were reached, the cleaning itself is in the ```common.py``` file, so nothing
needs to be run before step 3.

## Disclosure

- **Pretrained model:** TabPFN (Prior Labs), via the `tabpfn` package. Requires
  a licence and API token. It performs in-context learning and fits no
  parameters to this dataset.
- **AI coding agent:** AI agents were used to help with the technical implementation of the code
- **External solutions consulted:** the 1st, 2nd and 3rd place write-ups from
  the AMEX Default Prediction competition. Five techniques were tested from them which are DART boosting, `min_data_in_leaf`, `feature_fraction_bynode`,
  recency-window aggregates, within-customer ranks. **None were used in the final submission** as they all measured worse under cross-validation.

