"""Shared data loading and feature engineering for the default-probability models.

Both lgbm_model.py (LightGBM) and rejected/nn_model.py (MLP) build their matrices here so
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
MIN_PAY_RATE = 0.10
# Fold-split seeds to repeat CV over. Repeated CV is the one lever the OOF
# score cannot measure: each OOF row is predicted by its own fold's models
# only, while every test row is predicted by the average of all of them. The
# leaderboard confirmed this gap is real (OOF 0.4227 vs actual 0.4126).
REPEATS = tuple(range(6))


def clean(df):
    """Collapse the undocumented category codes onto 'other'.

    EDUCATION {0, 5, 6} -> 4 and MARRIAGE {0} -> 3. The data dictionary defines
    1-4 and 1-3 respectively; the extra codes cover a small tail and behave like
    'other'. 2_Column_Inspection.ipynb is where they were found, not where they
    are fixed -- this function is the only implementation, so train and test are
    treated identically.

    Idempotent, so it is safe to call on datasets/train_clean.csv, which already
    has it applied.
    """
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

    # Spending decomposition. The data gives balances and payments but never
    # the amount actually CHARGED, and those are different risk stories: a
    # balance rising because someone is spending is not the same as one rising
    # because they stopped paying. The accounting identity recovers it, since
    # PAY_AMT_t pays down BILL_AMT_(t+1):
    #     spend_t = BILL_t - BILL_(t+1) + PAY_AMT_t   (+ interest/fees)
    bill_m = X[BILL].to_numpy(float)
    amt_m = X[AMT].to_numpy(float)
    spend = bill_m[:, :5] - bill_m[:, 1:] + amt_m[:, :5]
    limv = lim.to_numpy()
    for i in range(5):
        X[f"spend{i + 1}"] = spend[:, i] / limv
    X["spend_mean"] = spend.mean(axis=1) / limv
    X["spend_max"] = spend.max(axis=1) / limv
    X["spend_std"] = spend.std(axis=1) / limv
    X["spend_trend"] = (spend[:, 0] - spend[:, 4]) / limv
    X["spend_total"] = spend.sum(axis=1) / limv
    X["n_months_no_spend"] = (spend <= 0).sum(axis=1)
    # charging more than repaying = debt accumulating under its own momentum
    paid_m = amt_m[:, :5]
    X["spend_minus_paid"] = (spend - paid_m).sum(axis=1) / limv
    X["months_spend_gt_paid"] = (spend > paid_m).sum(axis=1)

    # Minimum-payment behaviour. Issuers here required roughly 10% minimum, and
    # "pays the minimum and nothing more, every month" is a distress pattern the
    # payratio features cannot express -- they score that customer as merely
    # "low ratio", the same as someone who paid an arbitrary small amount.
    prev_m = bill_m[:, 1:]
    min_due = np.where(prev_m > 0, prev_m * MIN_PAY_RATE, np.nan)
    ratio_m = np.where(np.isnan(min_due) | (min_due == 0), np.nan, paid_m / min_due)
    with np.errstate(invalid="ignore"):
        X["min_pay_ratio_mean"] = np.nanmean(ratio_m, axis=1)
        X["min_pay_ratio_min"] = np.nanmin(np.where(np.isnan(ratio_m), np.inf, ratio_m), axis=1)
        X["months_paid_about_min"] = np.nansum((ratio_m >= 0.8) & (ratio_m <= 1.5), axis=1)
        X["months_paid_under_min"] = np.nansum(ratio_m < 0.8, axis=1)
    X["min_pay_ratio_min"] = X["min_pay_ratio_min"].replace(np.inf, np.nan)

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


def folds(X, y, seed=SEED):
    """The CV split every model uses, so their OOF vectors line up.

    seed selects which split. Models loop over REPEATS of these so the test
    predictions average across several different partitions, not just several
    seeds within one -- varying the partition decorrelates the ensemble more
    than re-seeding a fixed one does.
    """
    return StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)


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
