"""LightGBM default-probability model.

5-fold stratified CV reports AUC and log loss, then refits on all of train and
writes .output/predictions_lgbm.csv plus the raw OOF/test vectors for blending.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from common import folds, load, save, score

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
    X, y, X_test, ids = load()

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    rounds = []
    splits = list(folds(X, y))
    n_models = len(splits) * len(SEEDS)

    for fold, (tr, va) in enumerate(splits, 1):
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

    print()
    score("lgbm OOF", y, oof)

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

    save("lgbm", oof, preds, ids)
    print(f"base rate {y.mean():.4f}  mean predicted {preds.mean():.4f}")
    imp = pd.Series(full.feature_importances_, index=X.columns).nlargest(15)
    print("\ntop features\n" + imp.to_string())


if __name__ == "__main__":
    main()
