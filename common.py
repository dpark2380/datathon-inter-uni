"""Shared data loading and feature engineering for the default-probability models.

Both model.py (LightGBM) and nn_model.py (MLP) build their matrices here so
their out-of-fold predictions use identical rows, columns and CV folds and can
be blended directly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

OUT = Path(".output")
ID = "client_id"
TARGET = "default"
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
AMT = [f"PAY_AMT{i}" for i in range(1, 7)]
CATS = ["SEX", "EDUCATION", "MARRIAGE"]
SEED = 0


def clean(df):
    """Same code collapsing as Data Cleaning.ipynb: unknown codes -> 'other'."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def features(df):
    X = df.drop(columns=[TARGET], errors="ignore").copy()
    lim = X["LIMIT_BAL"]

    # Repayment-status history. PAY_0 is the most recent month.
    late = X[PAY].clip(lower=0)
    X["pay_max"] = late.max(axis=1)
    X["pay_sum"] = late.sum(axis=1)
    X["pay_mean"] = late.mean(axis=1)
    X["pay_std"] = X[PAY].std(axis=1)
    X["n_late"] = (late > 0).sum(axis=1)
    X["n_late2plus"] = (late >= 2).sum(axis=1)
    X["trend_pay"] = late["PAY_0"] - late["PAY_6"]
    X["recent_late"] = late[["PAY_0", "PAY_2"]].mean(axis=1)
    X["ever_paid_full"] = (X[PAY] == -1).any(axis=1).astype(int)
    X["never_used"] = (X[PAY] == -2).all(axis=1).astype(int)
    # Longest run of consecutive late months, most recent month first.
    run = np.zeros(len(X), dtype=int)
    best = np.zeros(len(X), dtype=int)
    for c in PAY:
        islate = (X[c] > 0).to_numpy()
        run = np.where(islate, run + 1, 0)
        best = np.maximum(best, run)
    X["max_late_streak"] = best
    X["months_since_late"] = (
        late.gt(0).to_numpy().argmax(axis=1).astype(float)
        + np.where(late.gt(0).any(axis=1), 0.0, np.nan)
    )
    X["months_since_late"] = X["months_since_late"].fillna(6)

    # Credit utilization.
    for i, c in enumerate(BILL, 1):
        X[f"util{i}"] = X[c] / lim
    utils = X[[f"util{i}" for i in range(1, 7)]]
    X["util_mean"] = utils.mean(axis=1)
    X["util_max"] = utils.max(axis=1)
    X["util_trend"] = X["util1"] - X["util6"]
    X["avail_credit"] = lim - X["BILL_AMT1"]

    # Payment coverage: PAY_AMTi pays down BILL_AMT(i+1), the prior month's bill.
    for i in range(1, 6):
        prev_bill = X[f"BILL_AMT{i + 1}"]
        X[f"payratio{i}"] = np.where(
            prev_bill > 0, X[f"PAY_AMT{i}"] / prev_bill.replace(0, np.nan), np.nan
        )
    ratios = X[[f"payratio{i}" for i in range(1, 6)]]
    X["payratio_mean"] = ratios.mean(axis=1)
    X["payratio_min"] = ratios.min(axis=1)
    X["paid_full_months"] = (ratios >= 0.99).sum(axis=1)
    X["n_zero_pay"] = (X[AMT] == 0).sum(axis=1)

    # Absolute levels and momentum.
    X["bill_sum"] = X[BILL].sum(axis=1)
    X["bill_mean"] = X[BILL].mean(axis=1)
    X["bill_std"] = X[BILL].std(axis=1)
    X["amt_sum"] = X[AMT].sum(axis=1)
    X["amt_mean"] = X[AMT].mean(axis=1)
    X["amt_std"] = X[AMT].std(axis=1)
    X["coverage_total"] = X["amt_sum"] / X["bill_sum"].replace(0, np.nan)
    X["bill_growth"] = X["BILL_AMT1"] - X["BILL_AMT6"]
    X["amt_over_limit"] = X["amt_sum"] / lim
    X["log_limit"] = np.log1p(lim)

    for c in CATS:
        X[c] = X[c].astype("category")
    return X


def load():
    """Cleaned train/test matrices with identical columns, plus target and ids."""
    train = clean(pd.read_csv("datasets/train_clean.csv", dtype={ID: str}))
    test = clean(pd.read_csv("datasets/test.csv", dtype={ID: str}))
    X = features(train.drop(columns=[ID]))
    X_test = features(test.drop(columns=[ID]))[X.columns]
    return X, train[TARGET].to_numpy(), X_test, test[ID]


def folds(X, y):
    """The one CV split both models use, so their OOF vectors line up."""
    return StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)


def score(name, y, p):
    print(f"{name:<14} auc {roc_auc_score(y, p):.5f}  log loss {log_loss(y, p):.5f}")


def save(name, oof, test_pred, ids):
    """Write predictions plus the raw vectors, so blends need no retraining."""
    OUT.mkdir(exist_ok=True)
    out = pd.DataFrame({ID: ids, "prob_default": test_pred})
    assert len(out) == len(ids) and out["prob_default"].between(0, 1).all()
    out.to_csv(OUT / f"predictions_{name}.csv", index=False)
    np.save(OUT / f"oof_{name}.npy", oof)
    np.save(OUT / f"test_{name}.npy", test_pred)
