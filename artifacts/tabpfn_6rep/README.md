# TabPFN 6-repeat + refit — the configuration behind the 0.40982 submission

`submission_tabpfn6.csv` (leaderboard log loss **0.40982**, our best) uses TabPFN
run at **6 fold-partitions with a 3-seed full-data refit**, blended
0.45 LightGBM / 0.30 GRU / 0.25 TabPFN.

`tabpfn_model_6rep.py` here is identical to `tabpfn_model.py` in the repo root
at the time of submission, kept alongside its outputs so the pairing is
unambiguous.

| File | What it is |
|---|---|
| `tabpfn_model_6rep.py` | The exact script that generated the vectors |
| `oof_tabpfn_6rep.npy` | Out-of-fold predictions, 24,000 rows (OOF log loss 0.42345) |
| `test_tabpfn_6rep.npy` | Test predictions, 6,000 rows (mean 0.2154) |
| `../test_lgbm_full.npy` | LightGBM test predictions (pure full-data refit) |
| `../test_seq.npy` | GRU test predictions |

## Reproducing submission_tabpfn6.csv without retraining

```python
import numpy as np, pandas as pd
lgbm = np.load("artifacts/test_lgbm_full.npy")     # pure full-data refit, NOT test_lgbm.npy
seq  = np.load("artifacts/test_seq.npy")
tab  = np.load("artifacts/tabpfn_6rep/test_tabpfn_6rep.npy")
preds = np.clip(0.45*lgbm + 0.30*seq + 0.25*tab, 1e-4, 1-1e-4)
ids = pd.read_csv("datasets/test.csv", usecols=["client_id"], dtype={"client_id": str})["client_id"]
pd.DataFrame({"client_id": ids, "prob_default": preds}).to_csv("submission_tabpfn6.csv", index=False)
```

Note the LightGBM component is the **pure full-data refit** (`test_lgbm_full.npy`),
not the 50/50 fold/refit mix that `model.py` writes to `test_lgbm.npy`. Using the
latter produces a different, slightly worse file (0.41038 vs 0.41032 when that
change was tested in isolation).

## Retraining from scratch

```bash
uv run python model.py                                  # ~10 min, writes both lgbm components
uv run python seq_model.py                              # ~50 min, GRU + refit
TABPFN_TOKEN="<token>" uv run --with tabpfn python tabpfn_model.py   # ~3h 10m
uv run python blend.py                                  # weight search + submission
```

TabPFN is a **pretrained model** (Prior Labs) requiring a licence and API token;
it performs in-context learning and fits no parameters to this dataset.
