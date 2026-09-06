"""Subgroup performance and calibration audit.

The competition brief asks teams to consider "responsible use of demographic
information". This reports, for each demographic subgroup, how well the model
performs and whether its probabilities mean the same thing.

Three questions, which are different and often confused:
  1. Does the model PREDICT more risk for some groups?  (predicted rate)
     -- a difference here may simply reflect a real difference in outcomes.
  2. Is it as ACCURATE for some groups as others?       (log loss, AUC)
     -- worse accuracy for a group means worse decisions for those people.
  3. Do its probabilities MEAN the same thing per group? (calibration ratio)
     -- this is the one that matters most for a lender. If "20% risk" really
     means 30% for one group and 15% for another, the same threshold applies
     a harsher standard to the first group even though the number is identical.

Uses the out-of-fold predictions from the shipped blend, so these are honest
estimates on unseen rows. Reads .output/oof_*.npy -- run the models first.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from common import ID, OUT, TARGET, clean

WEIGHTS = {"lgbm": 0.45, "seq": 0.30, "tabpfn": 0.25}
SEX = {1: "male", 2: "female"}
EDU = {1: "graduate school", 2: "university", 3: "high school", 4: "other"}
MAR = {1: "married", 2: "single", 3: "other"}


def audit(df, col, labels, p, y):
    rows = []
    for code, name in labels.items():
        m = (df[col] == code).to_numpy()
        if m.sum() < 30:
            continue
        actual, pred = y[m].mean(), p[m].mean()
        rows.append({
            "group": name,
            "n": int(m.sum()),
            "actual default rate": actual,
            "mean predicted": pred,
            "calibration ratio": pred / actual if actual > 0 else np.nan,
            "log loss": log_loss(y[m], np.clip(p[m], 1e-7, 1 - 1e-7), labels=[0, 1]),
            "auc": roc_auc_score(y[m], p[m]) if 0 < y[m].mean() < 1 else np.nan,
        })
    out = pd.DataFrame(rows)
    print(f"\n=== {col} ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if len(out) > 1:
        print(f"  spread in log loss      {out['log loss'].max() - out['log loss'].min():.4f}")
        print(f"  spread in calibration   {out['calibration ratio'].max() - out['calibration ratio'].min():.4f}"
              "   (0 = probabilities mean the same thing for every group)")
    return out


def main():
    raw = clean(pd.read_csv("datasets/train_clean.csv", dtype={ID: str}))
    y = raw[TARGET].to_numpy()
    p = sum(w * np.load(OUT / f"oof_{m}.npy") for m, w in WEIGHTS.items())

    print(f"blend OOF log loss {log_loss(y, p):.5f}   base rate {y.mean():.4f}   n = {len(y)}")
    print("\nCalibration ratio = mean predicted / actual rate.")
    print("1.00 is perfect; above 1 over-states risk for that group, below 1 under-states it.")

    for col, labels in (("SEX", SEX), ("EDUCATION", EDU), ("MARRIAGE", MAR)):
        audit(raw, col, labels, p, y)

    # age is continuous -- bucket it
    raw["age_band"] = pd.cut(raw["AGE"], [0, 25, 30, 35, 40, 50, 60, 200],
                             labels=["<=25", "26-30", "31-35", "36-40", "41-50", "51-60", "60+"])
    rows = []
    for band in raw["age_band"].cat.categories:
        m = (raw["age_band"] == band).to_numpy()
        if m.sum() < 30:
            continue
        actual, pred = y[m].mean(), p[m].mean()
        rows.append({"group": band, "n": int(m.sum()), "actual default rate": actual,
                     "mean predicted": pred, "calibration ratio": pred / actual,
                     "log loss": log_loss(y[m], np.clip(p[m], 1e-7, 1 - 1e-7), labels=[0, 1]),
                     "auc": roc_auc_score(y[m], p[m])})
    out = pd.DataFrame(rows)
    print("\n=== AGE ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"  spread in log loss      {out['log loss'].max() - out['log loss'].min():.4f}")
    print(f"  spread in calibration   {out['calibration ratio'].max() - out['calibration ratio'].min():.4f}")


if __name__ == "__main__":
    main()
