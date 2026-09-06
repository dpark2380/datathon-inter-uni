# TabPFN 3-repeat vectors — the configuration behind the 0.40995 submission

`submission_tabpfn3.csv` (leaderboard log loss **0.40995**) was produced with
TabPFN run at **3 fold-partitions and no full-data refit**. `tabpfn_model.py`
in the repo root was subsequently changed to 6 partitions plus a refit, so it
no longer reproduces that file. These artefacts preserve the original.

| File | What it is |
|---|---|
| `tabpfn_model_3rep.py` | The exact script that generated the vectors below |
| `oof_tabpfn_3rep.npy` | Out-of-fold predictions, 24,000 rows (OOF log loss 0.42360) |
| `test_tabpfn_3rep.npy` | Test predictions, 6,000 rows (mean 0.2060) |

## Reproducing submission_tabpfn3.csv without retraining

Blend weights 0.5 / 0.3 / 0.2 over LightGBM, GRU and TabPFN:

```python
import numpy as np, pandas as pd
lgbm = np.load(".output/test_lgbm_full.npy")   # pure full-data refit
seq  = np.load(".output/test_seq.npy")
tab  = np.load("artifacts/tabpfn_3rep/test_tabpfn_3rep.npy")
preds = np.clip(0.5*lgbm + 0.3*seq + 0.2*tab, 1e-4, 1-1e-4)
ids = pd.read_csv("datasets/test.csv", usecols=["client_id"], dtype={"client_id": str})["client_id"]
pd.DataFrame({"client_id": ids, "prob_default": preds}).to_csv("submission_tabpfn3.csv", index=False)
```

## Retraining from scratch

Requires a Prior Labs licence and API token (`TABPFN_TOKEN`), since TabPFN
downloads pretrained weights:

```bash
TABPFN_TOKEN="<token>" uv run --with tabpfn python artifacts/tabpfn_3rep/tabpfn_model_3rep.py
```

Runtime ~50 minutes on CPU. TabPFN is a **pretrained model** (Prior Labs),
not fitted to this dataset — it performs in-context learning from the training
rows supplied as context.
