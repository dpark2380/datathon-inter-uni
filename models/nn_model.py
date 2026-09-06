"""Neural network default-probability model.

A plain feed-forward net (sklearn MLPClassifier) on the same features as
model.py. Money columns are median-imputed and standardized and the three
categorical codes are one-hot encoded, because a net -- unlike LightGBM --
cannot split on raw scale or on a category code directly.

It loses to LightGBM on its own (0.7830 vs 0.7891 OOF auc), which is the usual
result on tabular data this size. It is worth training anyway: its errors are
decorrelated enough from the trees that blend.py gets a small lift from it.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CATS, folds, load, save, score

# A wider or deeper net does not help -- (128,64) and (256,128) both scored
# ~0.771 in the sweep. 24k rows is too few to feed anything bigger.
HIDDEN = (64, 32)
ALPHA = 1e-4
# Seed averaging matters far more than architecture here: one seed scores
# 0.7787, five averaged score 0.7830. Each net lands in a different local
# minimum, so averaging cancels most of that noise.
SEEDS = range(5)


def make_model(num, seed):
    prep = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    # add_indicator keeps "this ratio was undefined" as a signal:
                    # payratio is NaN exactly when the prior bill was <= 0.
                    SimpleImputer(strategy="median", add_indicator=True),
                    StandardScaler(),
                ),
                num,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATS),
        ]
    )
    net = MLPClassifier(
        hidden_layer_sizes=HIDDEN,
        alpha=ALPHA,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.12,
        random_state=seed,
    )
    return make_pipeline(prep, net)


def main():
    X, y, X_test, ids = load()
    num = [c for c in X.columns if c not in CATS]
    # Ratio features divide by bill amounts, which can be 0 or negative.
    for df in (X, X_test):
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    splits = list(folds(X, y))
    n_models = len(splits) * len(SEEDS)

    for fold, (tr, va) in enumerate(splits, 1):
        for seed in SEEDS:
            pipe = make_model(num, seed)
            pipe.fit(X.iloc[tr], y[tr])
            oof[va] += pipe.predict_proba(X.iloc[va])[:, 1] / len(SEEDS)
            test_pred += pipe.predict_proba(X_test)[:, 1] / n_models
        print(f"fold {fold}  auc {roc_auc_score(y[va], oof[va]):.5f}")

    print()
    score("mlp OOF", y, oof)
    save("nn", oof, test_pred, ids)
    print(f"base rate {y.mean():.4f}  mean predicted {test_pred.mean():.4f}")


if __name__ == "__main__":
    main()
