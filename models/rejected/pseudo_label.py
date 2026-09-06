"""Does soft pseudo-labeling the 6,000 unlabeled test rows help LightGBM?

Honest protocol -- the whole point of this file is that the answer must not be
contaminated:

  for each CV fold:
    1. fit a generator on an inner 90% of the training fold, early stopped on
       the inner 10%. The validation fold is never touched, not even through
       the choice of boosting rounds, so the pseudo-labels it produces carry
       no information about the rows we score on.
    2. soft-label the test rows with it: each test row enters the augmented
       training set twice, as label 1 with weight p and label 0 with weight
       1-p (LightGBM's binary objective needs 0/1 labels, and this is the
       weighted-duplicate encoding of a soft target), scaled by W.
    3. refit on train-fold + pseudo rows, early stopped on the validation fold
       exactly as lgbm_model.py does, and predict the validation fold.

  The baseline arm is the identical model without step 2, run on the same
  folds and seeds, so the difference is the pseudo-labels and nothing else.

RESULT (null): over 3 repeats, baseline OOF 0.42242 vs pseudo-labeled 0.42256
(w=0.25), 0.42259 (w=0.5), 0.42266 (w=1.0) -- every weight is worse, and worse
monotonically in how much the pseudo rows count. The unlabeled rows carry no
information the model does not already have; all soft-labeling does is make it
more confident about what it already believed. Differences are ~0.0002, inside
fold noise, so the honest statement is "no effect", not "harmful".

Run: uv run python pseudo_label.py [n_repeats]
"""

import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import sys
from pathlib import Path

# This script lives one level below the pipeline modules it imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import folds, load, save, score
from lgbm_model import PARAMS

WEIGHTS = (0.25, 0.5, 1.0)   # total weight given to the pseudo rows
SEED = 0


def fit(Xtr, ytr, wtr, Xva, yva, seed):
    m = lgb.LGBMClassifier(**PARAMS, random_state=seed)
    m.fit(Xtr, ytr, sample_weight=wtr, eval_X=Xva, eval_y=yva,
          eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(150, verbose=False)])
    return m


def main():
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    X, y, X_test, ids = load()

    base = np.zeros(len(X))
    pl = {w: np.zeros(len(X)) for w in WEIGHTS}
    pl_test = {w: np.zeros(len(X_test)) for w in WEIGHTS}
    n_folds = 5 * n_rep

    for rep in range(n_rep):
        for tr, va in folds(X, y, seed=rep):
            Xtr, ytr, Xva, yva = X.iloc[tr], y[tr], X.iloc[va], y[va]

            # 1. generator: inner split only, validation fold untouched
            inner_tr, inner_va = next(
                StratifiedKFold(10, shuffle=True, random_state=SEED).split(Xtr, ytr)
            )
            gen = fit(Xtr.iloc[inner_tr], ytr[inner_tr], None,
                      Xtr.iloc[inner_va], ytr[inner_va], SEED)
            p_test = gen.predict_proba(X_test)[:, 1]

            # baseline arm
            b = fit(Xtr, ytr, None, Xva, yva, SEED)
            base[va] += b.predict_proba(Xva)[:, 1] / n_rep

            # 2/3. soft pseudo-labels as weighted duplicates
            Xaug = pd.concat([Xtr, X_test, X_test], ignore_index=True)
            for c in X.columns:
                Xaug[c] = Xaug[c].astype(X[c].dtype)
            yaug = np.concatenate([ytr, np.ones(len(X_test)), np.zeros(len(X_test))])
            for w in WEIGHTS:
                waug = np.concatenate([
                    np.ones(len(Xtr)), w * p_test, w * (1 - p_test)
                ])
                m = fit(Xaug, yaug, waug, Xva, yva, SEED)
                pl[w][va] += m.predict_proba(Xva)[:, 1] / n_rep
                pl_test[w] += m.predict_proba(X_test)[:, 1] / n_folds
        print(f"repeat {rep} done", flush=True)

    clip = lambda p: np.clip(p, 1e-7, 1 - 1e-7)
    base = clip(base)
    print()
    score("baseline", y, base)
    best_w, best_ll = None, log_loss(y, base)
    for w in WEIGHTS:
        pl[w] = clip(pl[w])
        ll = log_loss(y, pl[w])
        print(f"pseudo w={w:<5} auc {roc_auc_score(y, pl[w]):.5f}  log loss {ll:.5f}"
              f"  ({ll - log_loss(y, base):+.5f} vs baseline)")
        if ll < best_ll:
            best_w, best_ll = w, ll

    if best_w is None:
        print("\nno pseudo-label weight beats the baseline -- null result")
    else:
        print(f"\nbest w={best_w} by {log_loss(y, base) - best_ll:.5f}"
              f"{'  (under 0.0005 = fold noise)' if log_loss(y, base) - best_ll < 5e-4 else ''}")
        save("pl", pl[best_w], clip(pl_test[best_w]), ids)


if __name__ == "__main__":
    main()
