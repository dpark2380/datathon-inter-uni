"""GRU sequence model over the 6-month repayment panel.

Everything else flattens the monthly history into summary stats (pay_max,
max_late_streak, util_trend...), which throws away event *order* --
"recovering after a bad patch" and "deteriorating" can share the same
mean/max/streak. This model instead reads the panel as a real time series
(6 timesteps x 8 channels, oldest month first) so the GRU can pick up
trajectories the aggregates destroy.

The engineered static features are concatenated onto the GRU's final hidden
state, so it sees both views rather than duplicating lgbm_model.py's feature
work.

Same 5-fold CV as everything else (via common.folds), so OOF lines up for
blending. Trained on BCE, which is the competition's own metric.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (AMT, BILL, CATS, ID, PAY, REPEATS, TARGET, clean, folds,
                    load, save, score)

# fewer seeds per repeat than before: with the same compute budget, varying the
# fold partition decorrelates the ensemble more than re-seeding a fixed one
SEEDS = tuple(range(5))
HIDDEN = 32
MAX_EPOCHS = 120
PATIENCE = 12
BATCH = 512
LR = 2e-3
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def panel(df):
    """(N, 6, C) tensor, timestep 0 = oldest month, 5 = most recent."""
    lim = df["LIMIT_BAL"].to_numpy()[:, None]
    pay = df[PAY].to_numpy(float)          # index 0 = most recent
    bill = df[BILL].to_numpy(float)
    amt = df[AMT].to_numpy(float)

    # month-over-month balance change; column i is the newer month, i+1 the older
    bill_delta = np.zeros_like(bill)
    bill_delta[:, :5] = (bill[:, :5] - bill[:, 1:]) / lim
    # payment t covers the previous month's bill
    coverage = np.zeros_like(amt)
    prev = bill[:, 1:]
    coverage[:, :5] = np.where(prev > 0, amt[:, :5] / np.where(prev == 0, np.nan, prev), 0.0)

    chans = [
        pay,
        (pay > 0).astype(float),
        bill / lim,
        amt / lim,
        np.sign(bill) * np.log1p(np.abs(bill)) / 10.0,
        np.log1p(np.abs(amt)) / 10.0,
        bill_delta,
        np.clip(coverage, -2, 2),
    ]
    # stack -> (N, 6, C), then reverse time so the GRU reads oldest -> newest
    x = np.stack(chans, axis=-1)[:, ::-1, :]
    return np.ascontiguousarray(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)


class Net(nn.Module):
    def __init__(self, n_static, n_chan=8, hidden=HIDDEN):
        super().__init__()
        # From a hyperparameter search: smaller hidden size + 2 layers + more
        # dropout beat the hand-picked hidden=64/dropout=0.3 single layer --
        # same "small and regularized wins at 24k rows" lesson as LightGBM.
        self.gru = nn.GRU(n_chan, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.4)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + n_static, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

    def forward(self, seq, static):
        _, h = self.gru(seq)
        h = torch.cat([h[-2], h[-1]], dim=1)  # both directions' final states
        return self.head(torch.cat([h, static], dim=1)).squeeze(1)


def fit_predict(seq_tr, st_tr, y_tr, seq_va, st_va, y_va, seq_te, st_te, seed,
                make_model=None):
    """Train one net with early stopping on validation log loss.

    make_model lets a different architecture reuse this loop unchanged --
    rejected/cnn_model.py passes its own factory rather than duplicating the training
    code.
    """
    torch.manual_seed(seed)
    make_model = make_model or (lambda n: Net(n))
    model = make_model(st_tr.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()

    t = lambda a: torch.tensor(a, device=DEV)
    seq_tr_t, st_tr_t, y_tr_t = t(seq_tr), t(st_tr), t(y_tr.astype(np.float32))
    seq_va_t, st_va_t = t(seq_va), t(st_va)
    seq_te_t, st_te_t = t(seq_te), t(st_te)

    best, best_state, bad = np.inf, None, 0
    n = len(seq_tr)
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            loss = lossf(model(seq_tr_t[idx], st_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(seq_va_t, st_va_t)).cpu().numpy()
        ll = log_loss(y_va, np.clip(p_va, 1e-7, 1 - 1e-7))
        if ll < best - 1e-5:
            best, bad = ll, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(model(seq_va_t, st_va_t)).cpu().numpy()
        p_te = torch.sigmoid(model(seq_te_t, st_te_t)).cpu().numpy()
    return p_va, p_te, epoch - bad + 1


def fit_full(seq_tr, st_tr, y_tr, seq_te, st_te, seed, epochs, make_model=None):
    """Refit on all training rows for a fixed epoch count.

    No validation set to early-stop on, so we reuse the average best epoch
    from CV (same trick lgbm_model.py uses with its average best boosting
    iteration). Fold models each saw 80% of the data; this one sees 100%,
    and the two get averaged for the test predictions.
    """
    torch.manual_seed(1000 + seed)
    make_model = make_model or (lambda n: Net(n))
    model = make_model(st_tr.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()

    t = lambda a: torch.tensor(a, device=DEV)
    seq_tr_t, st_tr_t, y_tr_t = t(seq_tr), t(st_tr), t(y_tr.astype(np.float32))
    n = len(seq_tr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            lossf(model(seq_tr_t[idx], st_tr_t[idx]), y_tr_t[idx]).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(t(seq_te), t(st_te))).cpu().numpy()


def preprocessors(X, seq_all, rows):
    """Fit the static-feature and sequence-channel transforms on `rows` only.

    Returns (static, norm): `static` scales + one-hot-encodes a feature frame
    for the head; `norm` standardises a panel tensor per channel. Fitting
    only on `rows` (a training fold during CV, or every row for the full
    refit) is what keeps validation/test data out of the fit -- one function
    for both cases means that leakage discipline lives in a single place.
    """
    num = [c for c in X.columns if c not in CATS]
    fit_on = X.iloc[rows]
    imp = SimpleImputer(strategy="median").fit(fit_on[num])
    sc = StandardScaler().fit(imp.transform(fit_on[num]))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(fit_on[CATS])

    chan = seq_all.shape[2]
    mu = seq_all[rows].reshape(-1, chan).mean(0)
    sd = seq_all[rows].reshape(-1, chan).std(0) + 1e-6

    def static(df):
        return np.hstack([sc.transform(imp.transform(df[num])),
                          ohe.transform(df[CATS])]).astype(np.float32)

    def norm(panel_tensor):
        return ((panel_tensor - mu) / sd).astype(np.float32)

    return static, norm


def main():
    train_raw = clean(pd.read_csv("datasets/train_clean.csv", dtype={ID: str}))
    test_raw = clean(pd.read_csv("datasets/test.csv", dtype={ID: str}))
    X, y, X_test, ids = load()

    seq_all, seq_test = panel(train_raw), panel(test_raw)
    # ratio features divide by bill amounts, which can be zero or negative
    num = [c for c in X.columns if c not in CATS]
    for df in (X, X_test):
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    n_models = 5 * len(SEEDS) * len(REPEATS)
    epochs_used = []

    for rep in REPEATS:
        for fold, (tr, va) in enumerate(folds(X, y, seed=rep), 1):
            static, norm = preprocessors(X, seq_all, tr)      # fitted on the fold only
            st_tr, st_va, st_te = static(X.iloc[tr]), static(X.iloc[va]), static(X_test)

            for seed in SEEDS:
                p_va, p_te, ep = fit_predict(
                    norm(seq_all[tr]), st_tr, y[tr],
                    norm(seq_all[va]), st_va, y[va],
                    norm(seq_test), st_te, seed + 100 * rep,
                )
                # each repeat covers every row once, so this averages repeats
                oof[va] += p_va / (len(SEEDS) * len(REPEATS))
                test_pred += p_te / n_models
                epochs_used.append(ep)
        print(f"repeat {rep} done", flush=True)

    # float32 sigmoids summed in float64 can land a hair outside [0,1]
    oof = np.clip(oof, 1e-7, 1 - 1e-7)
    print()
    score("seq OOF", y, oof)

    # Refit on all 24k rows at the average best epoch, then average with the
    # fold ensemble. OOF can't measure this gain (it only sees 80%-data fold
    # models) but test predictions do, from a model trained on 25% more data
    # -- same rationale as lgbm_model.py's full refit.
    n_epochs = max(1, int(np.mean(epochs_used)))
    print(f"\nfull-data refit at {n_epochs} epochs (avg best epoch across CV)", flush=True)
    static, norm = preprocessors(X, seq_all, np.arange(len(X)))   # fitted on everything

    full_pred = np.zeros(len(X_test))
    for seed in SEEDS:
        full_pred += fit_full(norm(seq_all), static(X), y,
                              norm(seq_test), static(X_test), seed, n_epochs) / len(SEEDS)

    test_pred = np.clip(0.5 * test_pred + 0.5 * full_pred, 1e-7, 1 - 1e-7)
    save("seq", oof, test_pred, ids)
    print(f"base rate {y.mean():.4f}  mean predicted {test_pred.mean():.4f}")

    lgbm_oof = np.load(".output/oof_lgbm.npy")
    print(f"corr with lgbm OOF {np.corrcoef(oof, lgbm_oof)[0, 1]:.4f}")


if __name__ == "__main__":
    main()
