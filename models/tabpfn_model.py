"""TabPFN over the same features, folds and protocol as everything else.

TabPFN is a transformer pretrained on synthetic tabular data: it does
in-context learning instead of gradient training, conditioning on the
training rows rather than fitting parameters to them. That's a
different-in-kind inductive bias from LightGBM/GRU, which is why it earns
blend weight instead of being redundant like the rejected CatBoost/CNN
(fold-0 pilot: 0.41362 vs lgbm 0.41306 / GRU 0.41390, corr 0.988/0.984).

balance_probabilities is off on purpose -- it would skew predictions away
from the true base rate, and calibration checks confirm the blend doesn't
need it.

Needs TABPFN_TOKEN in the environment.
"""
import numpy as np
from sklearn.metrics import log_loss
from tabpfn import TabPFNClassifier

from common import CATS, REPEATS, folds, load, save, score

import torch
# MPS (Apple GPU) is ~7x faster than CPU here (0.89 vs 6.3 min/fold) and
# gives identical predictions -- earlier runs assumed Metal wouldn't work
# and used CPU, untested.
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
REPS = REPEATS[:6]  # 6 partitions, ~16 min each
REFIT_SEEDS = (0, 1, 2)
# N_ESTIMATORS is TabPFN's own internal ensemble size. 8 looked better on a
# single fold but scored worse over the full 6-repeat CV (solo 0.42358 vs
# 0.42345, blend 0.42138 vs 0.42136) -- fold-level differences here are noise.
N_ESTIMATORS = 4


def prep(X):
    """TabPFN wants plain numerics; category dtype -> integer codes."""
    Z = X.copy()
    num = [c for c in Z.columns if c not in CATS]
    Z[num] = Z[num].replace([np.inf, -np.inf], np.nan)
    for c in CATS:
        Z[c] = Z[c].cat.codes.astype(float)
    return Z


def make_classifier(seed, categorical_features):
    """Build the fixed TabPFN configuration used for folds and full refits."""
    return TabPFNClassifier(
        device=DEV,
        ignore_pretraining_limits=True,
        n_estimators=N_ESTIMATORS,
        categorical_features_indices=categorical_features,
        balance_probabilities=False,
        random_state=seed,
    )


def main():
    X, y, X_test, ids = load()
    Xz, Xt = prep(X), prep(X_test)
    cat_idx = [Xz.columns.get_loc(c) for c in CATS]

    oof = np.zeros(len(Xz))
    test_pred = np.zeros(len(Xt))
    n_models = 5 * len(REPS)

    for rep in REPS:
        for fold, (tr, va) in enumerate(folds(Xz, y, seed=rep), 1):
            clf = make_classifier(rep, cat_idx)
            clf.fit(Xz.iloc[tr], y[tr])
            oof[va] += clf.predict_proba(Xz.iloc[va])[:, 1] / len(REPS)
            test_pred += clf.predict_proba(Xt)[:, 1] / n_models
            print(f"  rep {rep} fold {fold} done", flush=True)
        print(f"repeat {rep} complete", flush=True)

    # Full-data refit: fold models see 19,200 rows, this one sees all 24,000.
    # Matters more for TabPFN than the others since it's in-context learning
    # -- the training rows ARE its evidence, so more data means more context,
    # not just more gradient steps. Averaged 50/50 with the fold ensemble,
    # same as lgbm_model.py and seq_model.py.
    print("\nfull-data refit on all 24k rows", flush=True)
    full = np.zeros(len(Xt))
    for seed in REFIT_SEEDS:
        clf = make_classifier(1000 + seed, cat_idx)
        clf.fit(Xz, y)
        full += clf.predict_proba(Xt)[:, 1] / len(REFIT_SEEDS)
        print(f"  refit seed {seed} done", flush=True)

    oof = np.clip(oof, 1e-7, 1 - 1e-7)
    test_pred = np.clip(0.5 * test_pred + 0.5 * full, 1e-7, 1 - 1e-7)
    print()
    score("tabpfn OOF", y, oof)
    save("tabpfn", oof, test_pred, ids)

    lg, sq = np.load(".output/oof_lgbm.npy"), np.load(".output/oof_seq.npy")
    print(f"corr with lgbm {np.corrcoef(oof, lg)[0,1]:.4f}  with seq {np.corrcoef(oof, sq)[0,1]:.4f}")

    cur = log_loss(y, 0.65 * lg + 0.35 * sq)
    best = (None, 9.0)
    for a in np.arange(0, 1.001, 0.05):
        for b in np.arange(0, 1.001 - a, 0.05):
            ll = log_loss(y, a * lg + b * sq + (1 - a - b) * oof)
            if ll < best[1]:
                best = ((round(a, 2), round(b, 2), round(1 - a - b, 2)), ll)
    print(f"\ncurrent lgbm+seq {cur:.5f}")
    print(f"best 3-way (lgbm,seq,tabpfn) {best[0]}  {best[1]:.5f}   delta {cur-best[1]:+.5f}")


if __name__ == "__main__":
    main()
