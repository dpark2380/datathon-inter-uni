"""TabPFN over the same features, folds and protocol as everything else.

TabPFN is a transformer pretrained on millions of synthetic tabular datasets
that does in-context learning: the training rows are supplied as context and it
predicts from a learned prior, fitting no parameters to our data at all. That
is a different inductive bias in kind, not just a different architecture -- and
on the fold-0 pilot it scored 0.41362 against LightGBM's 0.41306 and the GRU's
0.41390, with correlations (0.988 / 0.984) in the band where the GRU earned
blend weight rather than the band where CatBoost and the CNN were rejected.

Probabilities are deliberately left unbalanced: balance_probabilities would
pull them off the true base rate, and three separate calibration checks have
confirmed this blend is already correctly calibrated.

Needs TABPFN_TOKEN in the environment.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from tabpfn import TabPFNClassifier

from common import CATS, REPEATS, folds, load, save, score

REPS = REPEATS[:6]          # 6 partitions; ~16 min each
N_ESTIMATORS = 4            # TabPFN's own internal ensembling


def prep(X):
    """TabPFN wants plain numerics; category dtype -> integer codes."""
    Z = X.copy()
    num = [c for c in Z.columns if c not in CATS]
    Z[num] = Z[num].replace([np.inf, -np.inf], np.nan)
    for c in CATS:
        Z[c] = Z[c].cat.codes.astype(float)
    return Z


def main():
    X, y, X_test, ids = load()
    Xz, Xt = prep(X), prep(X_test)
    cat_idx = [Xz.columns.get_loc(c) for c in CATS]

    oof = np.zeros(len(Xz))
    test_pred = np.zeros(len(Xt))
    n_models = 5 * len(REPS)

    for rep in REPS:
        for fold, (tr, va) in enumerate(folds(Xz, y, seed=rep), 1):
            clf = TabPFNClassifier(
                device="cpu",
                ignore_pretraining_limits=True,
                n_estimators=N_ESTIMATORS,
                categorical_features_indices=cat_idx,
                balance_probabilities=False,
                random_state=rep,
            )
            clf.fit(Xz.iloc[tr], y[tr])
            oof[va] += clf.predict_proba(Xz.iloc[va])[:, 1] / len(REPS)
            test_pred += clf.predict_proba(Xt)[:, 1] / n_models
            print(f"  rep {rep} fold {fold} done", flush=True)
        print(f"repeat {rep} complete", flush=True)

    # Full-data refit: the fold models each see 19,200 rows, this one sees all
    # 24,000. For TabPFN that matters more than for the others -- it does
    # in-context learning, so the training rows ARE the evidence it reasons
    # from, and a refit gives it 25% more context rather than more gradient
    # steps. LightGBM and the GRU both already have this; TabPFN did not.
    # Averaged 50/50 with the fold ensemble, matching model.py and seq_model.py.
    print("\nfull-data refit on all 24k rows", flush=True)
    full = np.zeros(len(Xt))
    REFIT_SEEDS = (0, 1, 2)
    for s_ in REFIT_SEEDS:
        clf = TabPFNClassifier(
            device="cpu",
            ignore_pretraining_limits=True,
            n_estimators=N_ESTIMATORS,
            categorical_features_indices=cat_idx,
            balance_probabilities=False,
            random_state=1000 + s_,
        )
        clf.fit(Xz, y)
        full += clf.predict_proba(Xt)[:, 1] / len(REFIT_SEEDS)
        print(f"  refit seed {s_} done", flush=True)

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
