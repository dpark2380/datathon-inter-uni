"""Fit the calibrated forest on all of train, write test predictions.

LEAF is set from the sweep in rf_test.py. Change it if your sweep picked
a different value.
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ID = "client_id"
TARGET = "default"
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
STATE = {-2: "no_use", -1: "paid_full", 0: "revolving"}
LEAF = 100


def features(df):
    status = df[PAY].apply(
        lambda s: s.map(STATE).fillna("late_" + s.clip(upper=4).astype(int).astype(str))
    )
    late = df[PAY].clip(lower=0)
    seq = pd.DataFrame(index=df.index)
    seq["max_late"] = late.max(axis=1)
    seq["n_late"] = (late > 0).sum(axis=1)
    seq["mean_late"] = late.mean(axis=1)
    seq["trend"] = late["PAY_0"] - late["PAY_6"]
    seq["ever_paid_full"] = (df[PAY] == -1).any(axis=1).astype(int)
    seq["never_used"] = (df[PAY] == -2).all(axis=1).astype(int)
    return pd.concat([pd.get_dummies(status, drop_first=True), seq], axis=1)


train = pd.read_csv("datasets/train.csv", index_col=ID, dtype={ID: str})
test = pd.read_csv("datasets/test.csv", index_col=ID, dtype={ID: str})

X = features(train)
y = train[TARGET]
X_test = features(test).reindex(columns=X.columns, fill_value=0)

rf = CalibratedClassifierCV(
    RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=LEAF,
        max_features="sqrt",
        random_state=0,
        n_jobs=-1,
    ),
    method="isotonic",
    cv=5,
)
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

# The only honest comparison available. test.csv has no labels.
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for name, m in [("logistic", logit), ("rf calibrated", rf)]:
    s = -cross_val_score(m, X, y, cv=cv, scoring="neg_log_loss").mean()
    print(f"{name:<16} cv log loss  {s:.5f}")

rf.fit(X, y)
preds = pd.Series(rf.predict_proba(X_test)[:, 1], index=test.index, name="prob_default")

assert len(preds) == len(test), "row count changed"
print(f"\nbase rate {y.mean():.4f}, mean predicted {preds.mean():.4f}")
print(f"distinct predicted values {preds.nunique()}")
print(preds.head().to_string())

preds.to_csv("rf_predictions.csv")