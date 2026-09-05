"""Blend the LightGBM and MLP predictions.

Run model.py and nn_model.py first; this reads the vectors they saved, picks
the weight that maximises OOF auc, and writes .output/predictions_blend.csv.
The weight is chosen on the same 24k out-of-fold rows used to score the parts,
so it is a fair comparison, but the gain is small enough to be within fold
noise -- check the printout before preferring the blend over plain LightGBM.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import ID, OUT, TARGET, clean, score

GRID = np.round(np.arange(0.0, 0.55, 0.05), 2)


def main():
    y = clean(pd.read_csv("datasets/train_clean.csv", dtype={ID: str}))[TARGET].to_numpy()
    try:
        oof = {m: np.load(OUT / f"oof_{m}.npy") for m in ("lgbm", "nn")}
        test = {m: np.load(OUT / f"test_{m}.npy") for m in ("lgbm", "nn")}
    except FileNotFoundError as e:
        raise SystemExit(f"missing {e.filename} -- run model.py and nn_model.py first")

    for m in ("lgbm", "nn"):
        score(m, y, oof[m])
    print(f"\ncorrelation of the two OOF vectors {np.corrcoef(oof['lgbm'], oof['nn'])[0, 1]:.4f}")

    print("\nweight on nn:")
    aucs = {}
    for w in GRID:
        aucs[w] = roc_auc_score(y, (1 - w) * oof["lgbm"] + w * oof["nn"])
        print(f"  {w:.2f}  auc {aucs[w]:.5f}")

    best = max(aucs, key=aucs.get)
    gain = aucs[best] - aucs[0.0]
    print(f"\nbest weight {best:.2f}  auc {aucs[best]:.5f}  (+{gain:.5f} over lgbm alone)")
    if gain < 0.001:
        print("gain is under 0.001, i.e. inside fold noise -- plain lgbm is the safer pick")

    preds = (1 - best) * test["lgbm"] + best * test["nn"]
    ids = pd.read_csv("datasets/test.csv", usecols=[ID], dtype={ID: str})[ID]
    out = pd.DataFrame({ID: ids, "prob_default": preds})
    assert len(out) == len(ids) and out["prob_default"].between(0, 1).all()
    out.to_csv(OUT / "predictions_blend.csv", index=False)
    print(f"wrote {OUT / 'predictions_blend.csv'}  mean predicted {preds.mean():.4f}")


if __name__ == "__main__":
    main()
