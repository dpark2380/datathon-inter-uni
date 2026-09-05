"""Default-probability model for datasets/test.csv.

LightGBM on engineered repayment/bill/payment features. 5-fold stratified CV
reports AUC and log loss, then refits on all of train and writes
predictions.csv with one probability per test client_id.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

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


# Picked by a 5-fold sweep over depth/leaves/regularization. Shallow and
# heavily regularized wins: the signal here is mostly the PAY_* history and
# deeper trees just overfit it (deeper: 0.7843 auc, this: 0.7891).
PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=12,
    max_depth=4,
    min_child_samples=100,
    feature_fraction=0.5,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_lambda=30.0,
    n_estimators=3000,
    verbose=-1,
    n_jobs=-1,
)
SEEDS = (0, 1, 2)


def main():
    train = clean(pd.read_csv("datasets/train_clean.csv", dtype={ID: str}))
    test = clean(pd.read_csv("datasets/test.csv", dtype={ID: str}))

    ids = test[ID]
    y = train[TARGET].to_numpy()
    X = features(train.drop(columns=[ID]))
    X_test = features(test.drop(columns=[ID]))[X.columns]

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    rounds = []

    n_models = cv.get_n_splits() * len(SEEDS)
    for fold, (tr, va) in enumerate(cv.split(X, y), 1):
        for seed in SEEDS:
            m = lgb.LGBMClassifier(**PARAMS, random_state=seed)
            m.fit(
                X.iloc[tr],
                y[tr],
                eval_X=X.iloc[va],
                eval_y=y[va],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(150, verbose=False)],
            )
            oof[va] += m.predict_proba(X.iloc[va])[:, 1] / len(SEEDS)
            test_pred += m.predict_proba(X_test)[:, 1] / n_models
            rounds.append(m.best_iteration_ or PARAMS["n_estimators"])
        print(f"fold {fold}  auc {roc_auc_score(y[va], oof[va]):.5f}")

    print(f"\nOOF auc      {roc_auc_score(y, oof):.5f}")
    print(f"OOF log loss {log_loss(y, oof):.5f}")

    # Refit on all data at the average best iteration, then blend with the
    # fold ensemble; both are unbiased, averaging cuts variance.
    full_pred = np.zeros(len(X_test))
    for seed in SEEDS:
        full = lgb.LGBMClassifier(
            **{**PARAMS, "n_estimators": int(np.mean(rounds))}, random_state=seed
        )
        full.fit(X, y)
        full_pred += full.predict_proba(X_test)[:, 1] / len(SEEDS)
    preds = 0.5 * test_pred + 0.5 * full_pred

    out = pd.DataFrame({ID: ids, "prob_default": preds})
    assert len(out) == len(test) and out["prob_default"].between(0, 1).all()
    out.to_csv("predictions.csv", index=False)
    print(f"\nbase rate {y.mean():.4f}  mean predicted {preds.mean():.4f}")

    imp = pd.Series(full.feature_importances_, index=X.columns).nlargest(15)
    print("\ntop features\n" + imp.to_string())


if __name__ == "__main__":
    main()
