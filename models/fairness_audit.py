"""Fairness audit: does the shipped blend treat demographic subgroups fairly?

Run after lgbm_model.py, seq_model.py and tabpfn_model.py. Reads their
saved out-of-fold predictions, so results are honest on unseen rows. Prints
one table per group (SEX, EDUCATION, MARRIAGE, AGE band): group size,
default rate, mean predicted risk, calibration ratio, skill, and AUC.

Why: the competition brief asks teams to consider "responsible use of
demographic information". That splits into three different questions,
easily confused:

  1. Does it PREDICT more risk for some groups?   (mean predicted)
     A gap here may simply reflect a real difference in outcomes.

  2. Do its probabilities MEAN the same thing?    (calib ratio)
     The one that matters most for a lender: if "20% risk" is really 30% for
     one group and 15% for another, a single threshold applies a harsher
     standard to the first group even though the number is identical.

  3. Does it RANK as well within each group?      (auc, skill)
     Worse ranking means worse decisions for those people.

Raw log loss can't answer #3 on its own: it isn't comparable across groups
with different base rates (a 7%-default group has a lower achievable loss
than a 25% one, regardless of model quality; EDUCATION="other" has the
best raw loss here despite the worst AUC). `skill` fixes this: each group's
loss divided by the loss of predicting that group's own base rate. Below 1
beats that baseline; closer to 0 is better. Compare groups on `skill` and
`auc`, never on raw log loss.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from common import ID, OUT, TARGET, clean

# Must match blend.py's MODELS and the weights it selects.
WEIGHTS = {"lgbm": 0.45, "seq": 0.30, "tabpfn": 0.25}
MIN_ROWS = 30

SEX = {1: "male", 2: "female"}
EDU = {1: "graduate school", 2: "university", 3: "high school", 4: "other"}
MAR = {1: "married", 2: "single", 3: "other"}


def audit(name, groups, p, y):
    """groups: dict of label -> boolean mask over all rows."""
    rows, skipped = [], []
    for label, m in groups.items():
        n = int(m.sum())
        if n < MIN_ROWS or not 0 < y[m].mean() < 1:
            # Never drop a group silently: in a fairness audit, the smallest
            # groups are most at risk of being under-served, and omitting
            # them would flatter the spread figures below.
            skipped.append((label, n))
            continue
        actual, pred = y[m].mean(), p[m].mean()
        loss = log_loss(y[m], np.clip(p[m], 1e-7, 1 - 1e-7), labels=[0, 1])
        # loss of predicting this group's own base rate for everyone in it
        base = log_loss(y[m], np.full(n, actual), labels=[0, 1])
        rows.append({
            "group": label,
            "n": n,
            "defaults": int(y[m].sum()),
            "actual rate": actual,
            "mean predicted": pred,
            "calib ratio": pred / actual,
            "skill": loss / base,
            "auc": roc_auc_score(y[m], p[m]),
        })

    out = pd.DataFrame(rows)
    print(f"\n=== {name} ===")
    if out.empty:
        print("  no group met the reporting threshold")
    else:
        print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if len(out) > 1:
        print(f"  spread in calibration ratio  {out['calib ratio'].max() - out['calib ratio'].min():.4f}"
              "   (0 = probabilities mean the same for every group)")
        print(f"  spread in auc                {out['auc'].max() - out['auc'].min():.4f}")
    for label, n in skipped:
        print(f"  [not reported] {label}: n={n} (below {MIN_ROWS} rows or single-class)")
    return out


def main():
    assert abs(sum(WEIGHTS.values()) - 1) < 1e-9, "blend weights must sum to 1"

    raw = clean(pd.read_csv("datasets/train_clean.csv", dtype={ID: str}))
    y = raw[TARGET].to_numpy()
    try:
        p = sum(w * np.load(OUT / f"oof_{m}.npy") for m, w in WEIGHTS.items())
    except FileNotFoundError as e:
        raise SystemExit(
            f"missing {e.filename}. Run lgbm_model.py, seq_model.py and tabpfn_model.py first"
        )

    print(f"blend OOF log loss {log_loss(y, p):.5f}   base rate {y.mean():.4f}   n = {len(y)}")
    print("\ncalib ratio = mean predicted / actual rate. 1.00 is perfect;")
    print("above 1 over-states risk for that group, below 1 under-states it.")
    print("skill = group log loss / loss of predicting that group's own base rate;")
    print("lower is better, and unlike raw log loss it is comparable across groups.")

    for name, col, labels in (("SEX", "SEX", SEX), ("EDUCATION", "EDUCATION", EDU),
                              ("MARRIAGE", "MARRIAGE", MAR)):
        audit(name, {lab: (raw[col] == code).to_numpy() for code, lab in labels.items()}, p, y)

    bands = pd.cut(raw["AGE"], [0, 25, 30, 35, 40, 50, 60, 200],
                   labels=["<=25", "26-30", "31-35", "36-40", "41-50", "51-60", "60+"])
    audit("AGE", {str(b): (bands == b).to_numpy() for b in bands.cat.categories}, p, y)

    print("\nSmall groups carry wide uncertainty: a calibration ratio built on a few")
    print("dozen default events is directionally useful but not precise. Read the")
    print("`defaults` column before quoting any single ratio.")


if __name__ == "__main__":
    main()
