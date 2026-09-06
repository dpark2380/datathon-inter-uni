"""Multi-task GRU: predict `default` jointly with next-month account state.

Problem reformulation, not a new algorithm. seq_model.py trains one head on one
target; 24k rows x 22% positives is roughly 5k bits of supervision for a net
with thousands of parameters, and that is what limits it. The panel itself
carries far more signal than the label does, so this model adds a
self-supervised auxiliary task: from the forward GRU state after month t,
predict what month t+1 looks like (late? how late? utilization? coverage?).

That is 5 extra supervised timesteps per customer on 4 channels, and it is
leak-free by construction -- the forward direction of a bidirectional GRU at
step t is a function of steps <= t only, so predicting step t+1 from it never
sees its own answer. The backward direction (which does see the future) feeds
only the default head, never the auxiliary head.

The bet is that a trunk forced to forecast the trajectory learns a better
representation of "deteriorating" than one fit to the default label alone.

Usage: .venv/bin/python multitask_model.py [--alpha 0.3] [--repeats 3] [--seeds 3]
"""

import argparse

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

from common import CATS, ID, clean, folds, load, save, score
from seq_model import DEV, panel

HIDDEN = 32
MAX_EPOCHS = 120
PATIENCE = 12
BATCH = 512
LR = 2e-3
EPS = 1e-7
# panel() channel order: pay, is_late, util, amt/lim, log bill, log amt,
# bill delta, coverage. Forecast the four that describe account health.
AUX_CH = [0, 1, 2, 7]
LATE_CH = 1  # index within AUX_CH is 1: binary, so BCE rather than MSE


class MultiNet(nn.Module):
    def __init__(self, n_static, n_chan=8, hidden=HIDDEN, n_aux=len(AUX_CH)):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(n_chan, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.4)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + n_static, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.15), nn.Linear(32, 1),
        )
        self.aux = nn.Linear(hidden, n_aux)

    def forward(self, seq, static):
        out, h = self.gru(seq)
        main = self.head(torch.cat([h[-2], h[-1], static], dim=1)).squeeze(1)
        # forward direction only: out[:, t, :hidden] depends on steps <= t
        aux = self.aux(out[:, :-1, : self.hidden])  # (N, T-1, n_aux)
        return main, aux


def aux_targets(seq):
    """Next-step target: what channels AUX_CH look like at t+1, for t=0..T-2."""
    return seq[:, 1:, AUX_CH]


def run_fold(seq_tr, st_tr, y_tr, seq_va, st_va, y_va, seq_te, st_te, seed, alpha,
             epochs=None):
    """Train one net. epochs=None -> early stop on validation *default* log loss."""
    torch.manual_seed(seed)
    model = MultiNet(st_tr.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    t = lambda a: torch.tensor(a, device=DEV)
    seq_tr_t, st_tr_t = t(seq_tr), t(st_tr)
    y_tr_t = t(y_tr.astype(np.float32))
    a_tr_t = t(aux_targets(seq_tr))
    # is_late is 0/1 before standardization; after it is not, so BCE needs the
    # raw indicator. Recover it by thresholding at the channel's own midpoint.
    late_tgt = (a_tr_t[:, :, LATE_CH] > a_tr_t[:, :, LATE_CH].mean()).float()
    other = [i for i in range(len(AUX_CH)) if i != LATE_CH]
    seq_va_t, st_va_t = t(seq_va), t(st_va)
    seq_te_t, st_te_t = t(seq_te), t(st_te)

    best, best_state, bad, n = np.inf, None, 0, len(seq_tr)
    last_epoch = 0
    for epoch in range(epochs or MAX_EPOCHS):
        last_epoch = epoch
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            main, aux = model(seq_tr_t[idx], st_tr_t[idx])
            loss = bce(main, y_tr_t[idx])
            loss = loss + alpha * (
                mse(aux[:, :, other], a_tr_t[idx][:, :, other])
                + bce(aux[:, :, LATE_CH], late_tgt[idx])
            )
            loss.backward()
            opt.step()

        if epochs is None:
            model.eval()
            with torch.no_grad():
                p_va = torch.sigmoid(model(seq_va_t, st_va_t)[0]).cpu().numpy()
            ll = log_loss(y_va, np.clip(p_va, EPS, 1 - EPS))
            if ll < best - 1e-5:
                best, bad, best_state = ll, 0, {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
            else:
                bad += 1
                if bad >= PATIENCE:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(model(seq_va_t, st_va_t)[0]).cpu().numpy()
        p_te = torch.sigmoid(model(seq_te_t, st_te_t)[0]).cpu().numpy()
    return p_va, p_te, last_epoch - bad + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.3, help="auxiliary loss weight")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--name", default="multitask")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    train_raw = clean(pd.read_csv("datasets/train_clean.csv", dtype={ID: str}))
    test_raw = clean(pd.read_csv("datasets/test.csv", dtype={ID: str}))
    X, y, X_test, ids = load()

    seq_all, seq_test = panel(train_raw), panel(test_raw)
    num = [c for c in X.columns if c not in CATS]
    for df in (X, X_test):
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    reps, seeds = range(a.repeats), range(a.seeds)
    n_models = 5 * len(seeds) * len(reps)
    epochs_used = []

    for rep in reps:
        for tr, va in folds(X, y, seed=rep):
            imp = SimpleImputer(strategy="median").fit(X.iloc[tr][num])
            sc = StandardScaler().fit(imp.transform(X.iloc[tr][num]))
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
                X.iloc[tr][CATS]
            )

            def static(df):
                return np.hstack([
                    sc.transform(imp.transform(df[num])), ohe.transform(df[CATS])
                ]).astype(np.float32)

            st_tr, st_va, st_te = static(X.iloc[tr]), static(X.iloc[va]), static(X_test)
            flat = seq_all[tr].reshape(-1, seq_all.shape[2])
            mu, sd = flat.mean(0), flat.std(0) + 1e-6
            norm = lambda arr: ((arr - mu) / sd).astype(np.float32)

            for seed in seeds:
                p_va, p_te, ep = run_fold(
                    norm(seq_all[tr]), st_tr, y[tr],
                    norm(seq_all[va]), st_va, y[va],
                    norm(seq_test), st_te, seed + 100 * rep, a.alpha,
                )
                oof[va] += p_va / (len(seeds) * len(reps))
                test_pred += p_te / n_models
                epochs_used.append(ep)
        print(f"repeat {rep} done", flush=True)

    oof = np.clip(oof, EPS, 1 - EPS)
    print()
    score(f"multitask a={a.alpha}", y, oof)

    lgbm, seq = np.load(".output/oof_lgbm.npy"), np.load(".output/oof_seq.npy")
    print(f"corr lgbm {np.corrcoef(oof, lgbm)[0, 1]:.4f}  "
          f"corr seq {np.corrcoef(oof, seq)[0, 1]:.4f}")
    grid = np.round(np.arange(0, 1.001, 0.02), 2)
    cands = {(w1, w2): log_loss(y, w1 * lgbm + w2 * seq + (1 - w1 - w2) * oof)
             for w1 in grid for w2 in grid if w1 + w2 <= 1.0}
    (b1, b2), bll = min(cands.items(), key=lambda kv: kv[1])
    base = min(log_loss(y, w * lgbm + (1 - w) * seq) for w in grid)
    print(f"3-way best (lgbm {b1}, seq {b2}, mt {round(1 - b1 - b2, 2)}) "
          f"log loss {bll:.5f}  vs lgbm+seq {base:.5f}  gain {base - bll:.5f}")

    if a.no_save:
        return
    # full-data refit at the average best epoch, averaged with the fold ensemble
    n_epochs = max(1, int(np.mean(epochs_used)))
    imp = SimpleImputer(strategy="median").fit(X[num])
    sc = StandardScaler().fit(imp.transform(X[num]))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X[CATS])
    static_all = lambda df: np.hstack([
        sc.transform(imp.transform(df[num])), ohe.transform(df[CATS])
    ]).astype(np.float32)
    flat = seq_all.reshape(-1, seq_all.shape[2])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    nrm = lambda arr: ((arr - mu) / sd).astype(np.float32)

    full = np.zeros(len(X_test))
    for seed in seeds:
        _, p_te, _ = run_fold(
            nrm(seq_all), static_all(X), y, nrm(seq_all[:1]), static_all(X.iloc[:1]),
            y[:1], nrm(seq_test), static_all(X_test), 1000 + seed, a.alpha,
            epochs=n_epochs,
        )
        full += p_te / len(seeds)
    test_pred = np.clip(0.5 * test_pred + 0.5 * full, EPS, 1 - EPS)
    save(a.name, oof, test_pred, ids)
    print(f"saved {a.name}  base rate {y.mean():.4f}  mean pred {test_pred.mean():.4f}")


if __name__ == "__main__":
    main()
