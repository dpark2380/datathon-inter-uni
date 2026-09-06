"""GRU over the panel WITH spending as a sequence channel.

Spending decomposition is the only feature family that ever worked here
(+0.00046 OOF for LightGBM, -0.00173 on the leaderboard). But the GRU has only
ever seen it FLATTENED, as static features via common.load() -- never as a
per-month trajectory. This adds it to the sequence itself:

    spend_t = BILL_t - BILL_(t+1) + PAY_AMT_t

so the GRU can read the order of spending, not just its summary. Two new
channels: spending, and net debt change (spend minus payment) per month.

Identical to seq_model.py in every other respect, so the comparison against
its matched control isolates the channel change.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
from pathlib import Path

# This script lives one level below the pipeline modules it imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import AMT, BILL, CATS, ID, PAY, TARGET, clean, folds, load, save, score
import seq_model as S

REPS = (0, 1, 2)
SEEDS = (0, 1, 2)
N_CHAN = 10


def panel_spend(df):
    """(N, 6, 10): seq_model's 8 channels plus spending and net debt change.

    spend is only defined for t=1..5 (needs BILL_(t+1)); the oldest month is
    padded with 0 identically in train and test.
    """
    base = S.panel(df)                       # (N, 6, 8), already oldest-first
    lim = df["LIMIT_BAL"].to_numpy()[:, None]
    bill = df[BILL].to_numpy(float)          # col 0 = most recent
    amt = df[AMT].to_numpy(float)

    spend = np.zeros_like(bill)
    spend[:, :5] = bill[:, :5] - bill[:, 1:] + amt[:, :5]
    net = np.zeros_like(bill)
    net[:, :5] = spend[:, :5] - amt[:, :5]

    extra = np.stack([spend / lim, net / lim], axis=-1)[:, ::-1, :]  # oldest-first
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(np.concatenate([base, extra], axis=2), dtype=np.float32)


def run(panel_fn, n_chan, label):
    train_raw = clean(pd.read_csv("datasets/train_clean.csv", dtype={ID: str}))
    test_raw = clean(pd.read_csv("datasets/test.csv", dtype={ID: str}))
    X, y, X_test, ids = load()
    seq_all, seq_test = panel_fn(train_raw), panel_fn(test_raw)
    num = [c for c in X.columns if c not in CATS]
    for df in (X, X_test):
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    n_models = 5 * len(SEEDS) * len(REPS)
    mk = lambda n: S.Net(n, n_chan=n_chan)

    for rep in REPS:
        for tr, va in folds(X, y, seed=rep):
            imp = SimpleImputer(strategy="median").fit(X.iloc[tr][num])
            sc = StandardScaler().fit(imp.transform(X.iloc[tr][num]))
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X.iloc[tr][CATS])
            st = lambda d: np.hstack([sc.transform(imp.transform(d[num])),
                                      ohe.transform(d[CATS])]).astype(np.float32)
            st_tr, st_va, st_te = st(X.iloc[tr]), st(X.iloc[va]), st(X_test)
            mu = seq_all[tr].reshape(-1, n_chan).mean(0)
            sd = seq_all[tr].reshape(-1, n_chan).std(0) + 1e-6
            nz = lambda a: ((a - mu) / sd).astype(np.float32)
            for s in SEEDS:
                p_va, p_te, _ = S.fit_predict(nz(seq_all[tr]), st_tr, y[tr],
                                              nz(seq_all[va]), st_va, y[va],
                                              nz(seq_test), st_te, s + 100 * rep,
                                              make_model=mk)
                oof[va] += p_va / (len(SEEDS) * len(REPS))
                test_pred += p_te / n_models
        print(f"  {label}: repeat {rep} done", flush=True)

    oof = np.clip(oof, 1e-7, 1 - 1e-7)
    test_pred = np.clip(test_pred, 1e-7, 1 - 1e-7)
    print(f"{label:<22} log loss {log_loss(y, oof):.5f}  auc {roc_auc_score(y, oof):.5f}", flush=True)
    return oof, test_pred, ids, y


def main():
    ctrl_oof, _, _, y = run(S.panel, 8, "control (8 chan)")
    new_oof, new_test, ids, _ = run(panel_spend, N_CHAN, "+ spend channels")

    c, n = log_loss(y, ctrl_oof), log_loss(y, new_oof)
    print(f"\ndelta {c - n:+.5f}  (>0.0005 to count)")
    lg = np.load(".output/oof_lgbm.npy")
    print(f"corr with lgbm: control {np.corrcoef(ctrl_oof, lg)[0,1]:.4f}  new {np.corrcoef(new_oof, lg)[0,1]:.4f}")

    seq = np.load(".output/oof_seq.npy")
    cur = min(log_loss(y, (1 - w) * lg + w * seq) for w in np.arange(0, 1.001, 0.05))
    best = (None, 9)
    for a in np.arange(0, 1.001, 0.05):
        for b in np.arange(0, 1.001 - a, 0.05):
            ll = log_loss(y, a * lg + b * seq + (1 - a - b) * new_oof)
            if ll < best[1]:
                best = ((round(a, 2), round(b, 2), round(1 - a - b, 2)), ll)
    print(f"current lgbm+seq {cur:.5f}   best 3-way {best[0]} {best[1]:.5f}  delta {cur-best[1]:+.5f}")
    if n < c:
        save("seqspend", new_oof, new_test, ids)
        print("saved as seqspend")


if __name__ == "__main__":
    main()
