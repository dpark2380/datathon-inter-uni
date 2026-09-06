"""Discrete-time hazard reformulation of the default problem (Bai et al. 2019).

model.py asks one question per customer: "given six months of history, will they
default?" -- 24k labelled examples, and that label count is the binding
constraint. This file asks the same question at every month the panel supports:
roll the window back s months and ask "will they be late at month s?". Each
customer then contributes several rows, so the model fits the *hazard* of
deterioration as a function of recent history rather than one snapshot of it,
and the default label becomes the last step of that hazard sequence -- month 7's
event, the one month the issuer labelled for us.

Rolled-back rows carry the pseudo-target 1{PAY at month s >= 1}, whose base rate
(0.228 at month 1) nearly matches the default rate (0.221), so both tasks live
on the same scale and can share one model. A `shift` column marks the horizon a
row came from; prediction always happens at shift 0.

Leakage: a rolled-back row for customer i uses only months strictly older than
its own target month, rolled-back rows are added only for customers in the
training folds, and the pseudo-target is a function of the features rather than
of `default`, so it cannot carry the label.

Usage: .venv/bin/python survival_model.py [--shifts 2] [--aug-weight 0.5]
"""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from common import (AMT, BILL, CATS, ID, REPEATS, TARGET, clean, features,
                    folds, load, save, score)
from model import PARAMS

EPS = 1e-7
STATIC = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]


def paycol(m):
    """Column holding month m's repayment status (m=1 is the most recent)."""
    return "PAY_0" if m == 1 else f"PAY_{m}"


def rolled(raw, s):
    """Panel as it looked s months earlier, plus that month's late indicator.

    Months older than 6 do not exist, so the oldest available month is repeated
    to pad -- the same convention as carrying the earliest observation forward.
    """
    old = lambda m: min(m + s, 6)
    df = pd.DataFrame({c: raw[c].to_numpy() for c in STATIC})
    for m in range(1, 7):
        df[paycol(m)] = raw[paycol(old(m))].to_numpy()
        df[f"BILL_AMT{m}"] = raw[f"BILL_AMT{old(m)}"].to_numpy()
        df[f"PAY_AMT{m}"] = raw[f"PAY_AMT{old(m)}"].to_numpy()
    return df, (raw[paycol(s)] >= 1).astype(int).to_numpy()


def build(raw, cols, shifts):
    """Rolled-back feature matrices and pseudo-targets, one per shift."""
    out = []
    for s in shifts:
        df, y_s = rolled(raw, s)
        Xs = features(df).reindex(columns=cols)
        for c in CATS:
            Xs[c] = Xs[c].astype("category")
        Xs["shift"] = s
        out.append((Xs, y_s))
    return out


def run(X, y, X_test, aug, params, aux_weight, reps, seeds):
    """CV the augmented model. Returns OOF, test predictions, best iterations."""
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    n_models = 5 * len(seeds) * len(reps)
    rounds = []
    for rep in reps:
        for tr, va in folds(X, y, seed=rep):
            # rolled-back rows only for customers in the training folds
            X_tr = pd.concat([X.iloc[tr]] + [Xs.iloc[tr] for Xs, _ in aug], ignore_index=True)
            y_tr = np.concatenate([y[tr]] + [ys[tr] for _, ys in aug])
            w_tr = np.concatenate(
                [np.ones(len(tr))] + [np.full(len(tr), aux_weight)] * len(aug)
            )
            for c in CATS:
                X_tr[c] = X_tr[c].astype("category")
            for seed in seeds:
                m = lgb.LGBMClassifier(**params, random_state=seed + 100 * rep)
                m.fit(
                    X_tr, y_tr, sample_weight=w_tr,
                    eval_X=X.iloc[va], eval_y=y[va],
                    eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(150, verbose=False)],
                )
                oof[va] += m.predict_proba(X.iloc[va])[:, 1] / (len(seeds) * len(reps))
                test_pred += m.predict_proba(X_test)[:, 1] / n_models
                rounds.append(m.best_iteration_ or params["n_estimators"])
    return np.clip(oof, EPS, 1 - EPS), test_pred, rounds


def report(label, y, oof, rounds=None):
    """Solo score plus the number that actually decides things: 3-way blend."""
    lgbm, seq = np.load(".output/oof_lgbm.npy"), np.load(".output/oof_seq.npy")
    grid = np.round(np.arange(0, 1.001, 0.02), 2)
    cands = {(w1, w2): log_loss(y, w1 * lgbm + w2 * seq + (1 - w1 - w2) * oof)
             for w1 in grid for w2 in grid if w1 + w2 <= 1.0}
    (b1, b2), bll = min(cands.items(), key=lambda kv: kv[1])
    base = min(log_loss(y, w * lgbm + (1 - w) * seq) for w in grid)
    print(
        f"{label:<46} solo {log_loss(y, oof):.5f}  auc {roc_auc_score(y, oof):.5f}  "
        f"corr_lgbm {np.corrcoef(oof, lgbm)[0, 1]:.4f}  "
        f"3way {bll:.5f} (w={b1},{b2},{round(1 - b1 - b2, 2)}) gain {base - bll:+.5f}"
        + (f"  iters {int(np.mean(rounds))}" if rounds else ""),
        flush=True,
    )
    return bll


# Capacity grid for the AUGMENTED data. model.py's PARAMS were tuned on 24k
# rows; shifts=2 trains on ~72k, which usually supports more capacity.
CANDIDATES = {
    "baseline (model.py)": {},
    "leaves31 d6": dict(num_leaves=31, max_depth=6),
    "leaves31 d6 mcs50": dict(num_leaves=31, max_depth=6, min_child_samples=50),
    "leaves31 d6 lam10": dict(num_leaves=31, max_depth=6, reg_lambda=10.0),
    "leaves31 d6 lam100": dict(num_leaves=31, max_depth=6, reg_lambda=100.0),
    "leaves63 d8 mcs50": dict(num_leaves=63, max_depth=8, min_child_samples=50),
    "leaves63 d8 lam100": dict(num_leaves=63, max_depth=8, reg_lambda=100.0),
    "leaves24 d5": dict(num_leaves=24, max_depth=5),
    "d4 mcs30": dict(min_child_samples=30),
    "d4 mcs300 lam100": dict(min_child_samples=300, reg_lambda=100.0),
    "leaves31 d6 lr015 ff07": dict(num_leaves=31, max_depth=6, learning_rate=0.015,
                                   feature_fraction=0.7),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifts", type=int, default=2, help="how many months to roll back")
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--name", default="survival")
    ap.add_argument("--params", default="baseline (model.py)",
                    help=f"one of: {list(CANDIDATES)}")
    ap.add_argument("--sweep", choices=["params", "aug"],
                    help="params: capacity grid at fixed augmentation. "
                         "aug: shifts x weight grid at --params.")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    raw = clean(pd.read_csv("datasets/train_clean.csv", dtype={ID: str}))
    X, y, X_test, ids = load()
    cols = list(X.columns)
    X, X_test = X.copy(), X_test.copy()
    X["shift"], X_test["shift"] = 0, 0
    reps, seeds = range(a.repeats), range(a.seeds)

    # every panel we might need, built once and sliced per config
    panels = build(raw, cols, range(1, 5))
    print("pseudo-target rates by shift: "
          + ", ".join(f"s{i + 1} {ys.mean():.3f}" for i, (_, ys) in enumerate(panels)),
          flush=True)
    print(f"scoring {a.repeats} repeats x {a.seeds} seeds\n", flush=True)

    if a.sweep == "params":
        aug = panels[: a.shifts]
        for label, over in CANDIDATES.items():
            oof, _, rounds = run(X, y, X_test, aug, {**PARAMS, **over}, a.aux_weight,
                                 reps, seeds)
            report(label, y, oof, rounds)
        return
    if a.sweep == "aug":
        params = {**PARAMS, **CANDIDATES[a.params]}
        for s in (1, 2, 3, 4):
            for w in (0.25, 0.5, 1.0):
                oof, _, rounds = run(X, y, X_test, panels[:s], params, w, reps, seeds)
                report(f"shifts={s} w={w}", y, oof, rounds)
        return

    params = {**PARAMS, **CANDIDATES[a.params]}
    aug = panels[: a.shifts]
    oof, test_pred, rounds = run(X, y, X_test, aug, params, a.aux_weight, reps, seeds)
    report(f"{a.params} shifts={a.shifts} w={a.aux_weight}", y, oof, rounds)

    if a.no_save:
        return
    # full-data refit at the average best iteration, averaged with the fold ensemble
    X_full = pd.concat([X] + [Xs for Xs, _ in aug], ignore_index=True)
    y_full = np.concatenate([y] + [ys for _, ys in aug])
    w_full = np.concatenate([np.ones(len(X))] + [np.full(len(X), a.aux_weight)] * len(aug))
    for c in CATS:
        X_full[c] = X_full[c].astype("category")
    full = np.zeros(len(X_test))
    for seed in seeds:
        m = lgb.LGBMClassifier(**{**params, "n_estimators": int(np.mean(rounds))},
                               random_state=seed)
        m.fit(X_full, y_full, sample_weight=w_full)
        full += m.predict_proba(X_test)[:, 1] / len(seeds)
    test_pred = np.clip(0.5 * test_pred + 0.5 * full, EPS, 1 - EPS)
    save(a.name, oof, test_pred, ids)
    print(f"saved {a.name}  base rate {y.mean():.4f}  mean pred {test_pred.mean():.4f}")


if __name__ == "__main__":
    main()
