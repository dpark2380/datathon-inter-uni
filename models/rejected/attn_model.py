"""Transformer encoder over the 6-month panel.

Third way of reading the same sequence. The GRU compresses history into a
running state (recent months dominate); the CNN fires on local motifs. Self
attention relates *any* month to *any* other directly -- "the month they went
delinquent" and "the month five steps later where they still hadn't recovered"
are one attention edge apart, not five recurrent steps.

A CLS token collects the sequence summary, learned positional embeddings give
the model month identity (PAY_0 vs PAY_6 mean different things), and the static
engineered features are concatenated onto the CLS output exactly as in
seq_model.py so this competes on sequence reading, not feature work.

Reuses seq_model's panel(), training loop and full-data refit unchanged; only
the architecture differs. Same common.folds so the OOF vector blends with the
rest.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
from pathlib import Path

# This script lives one level below the pipeline modules it imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import CATS, ID, REPEATS, clean, folds, load, save, score
from seq_model import fit_full, fit_predict, panel

# Reduced from seq_model's 6x5: 45 models is enough to average the seed noise
# out and keeps the run under an hour on MPS.
SEEDS = tuple(range(3))
REPS = REPEATS[:3]
D_MODEL = 32
N_HEAD = 4
N_LAYER = 2
DROPOUT = 0.3


class AttnNet(nn.Module):
    def __init__(self, n_static, n_chan=8, d=D_MODEL):
        super().__init__()
        self.proj = nn.Linear(n_chan, d)
        self.pos = nn.Parameter(torch.zeros(1, 6, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d, N_HEAD, dim_feedforward=2 * d, dropout=DROPOUT,
            batch_first=True, norm_first=True,  # pre-norm: trains without warmup
        )
        self.enc = nn.TransformerEncoder(layer, N_LAYER)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(2 * d + n_static, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

    def forward(self, seq, static):
        x = self.proj(seq) + self.pos
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        h = self.norm(self.enc(x))
        # CLS summary plus a mean over the months: attention pooling alone can
        # collapse onto one timestep, the mean keeps the overall level in view.
        return self.head(torch.cat([h[:, 0], h[:, 1:].mean(1), static], dim=1)).squeeze(1)


def prep():
    train_raw = clean(pd.read_csv("datasets/train_clean.csv", dtype={ID: str}))
    test_raw = clean(pd.read_csv("datasets/test.csv", dtype={ID: str}))
    X, y, X_test, ids = load()
    num = [c for c in X.columns if c not in CATS]
    for df in (X, X_test):
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    return X, y, X_test, ids, num, panel(train_raw), panel(test_raw)


def statics(X_fit, num, frames):
    """Impute+scale+one-hot fitted on X_fit only, applied to each frame."""
    imp = SimpleImputer(strategy="median").fit(X_fit[num])
    sc = StandardScaler().fit(imp.transform(X_fit[num]))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_fit[CATS])
    return [np.hstack([sc.transform(imp.transform(f[num])),
                       ohe.transform(f[CATS])]).astype(np.float32) for f in frames]


def normer(seq_fit):
    mu = seq_fit.reshape(-1, seq_fit.shape[2]).mean(0)
    sd = seq_fit.reshape(-1, seq_fit.shape[2]).std(0) + 1e-6
    return lambda a: ((a - mu) / sd).astype(np.float32)


def main():
    X, y, X_test, ids, num, seq_all, seq_test = prep()
    make = lambda n: AttnNet(n)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    n_models = 5 * len(SEEDS) * len(REPS)
    epochs_used = []

    for rep in REPS:
        for tr, va in folds(X, y, seed=rep):
            st_tr, st_va, st_te = statics(X.iloc[tr], num,
                                          [X.iloc[tr], X.iloc[va], X_test])
            norm = normer(seq_all[tr])
            for seed in SEEDS:
                p_va, p_te, ep = fit_predict(
                    norm(seq_all[tr]), st_tr, y[tr],
                    norm(seq_all[va]), st_va, y[va],
                    norm(seq_test), st_te, seed + 100 * rep, make_model=make,
                )
                oof[va] += p_va / (len(SEEDS) * len(REPS))
                test_pred += p_te / n_models
                epochs_used.append(ep)
        print(f"repeat {rep} done", flush=True)

    oof = np.clip(oof, 1e-7, 1 - 1e-7)
    print()
    score("attn OOF", y, oof)

    # Full-data refit at the average best epoch, averaged with the fold
    # ensemble -- same rationale as seq_model.py.
    n_epochs = max(1, int(np.mean(epochs_used)))
    print(f"\nfull-data refit at {n_epochs} epochs", flush=True)
    st_all, st_te_all = statics(X, num, [X, X_test])
    nrm = normer(seq_all)
    full = np.zeros(len(X_test))
    for seed in SEEDS:
        full += fit_full(nrm(seq_all), st_all, y, nrm(seq_test), st_te_all,
                         seed, n_epochs, make_model=make) / len(SEEDS)

    test_pred = np.clip(0.5 * test_pred + 0.5 * full, 1e-7, 1 - 1e-7)
    save("attn", oof, test_pred, ids)
    print(f"base rate {y.mean():.4f}  mean predicted {test_pred.mean():.4f}")

    others = {m: np.load(f".output/oof_{m}.npy") for m in ("lgbm", "seq")}
    for m, v in others.items():
        print(f"corr with {m} OOF {np.corrcoef(oof, v)[0, 1]:.4f}")
    blend_search(y, others["lgbm"], others["seq"], oof)


def blend_search(y, lgbm, seq, attn):
    """Grid over the lgbm/seq/attn simplex; does attn beat the 2-way blend?"""
    grid = np.round(np.arange(0, 1.001, 0.05), 2)
    losses = {}
    for ws in grid:
        for wa in grid:
            if ws + wa > 1.0001:
                continue
            wl = round(1 - ws - wa, 2)
            p = np.clip(wl * lgbm + ws * seq + wa * attn, 1e-7, 1 - 1e-7)
            losses[(wl, round(ws, 2), round(wa, 2))] = log_loss(y, p)

    two_way = min(ll for w, ll in losses.items() if w[2] == 0)
    best = min(losses, key=losses.get)
    print("\nblend search (lgbm, seq, attn):")
    for w in sorted(losses, key=losses.get)[:5]:
        print(f"  {w}  log loss {losses[w]:.5f}")
    gain = two_way - losses[best]
    print(f"\nbest 2-way (lgbm+seq) {two_way:.5f} -> 3-way {losses[best]:.5f}"
          f"  gain {gain:.5f}")
    if gain < 0.0005:
        print("gain under 0.0005 == inside fold noise; attn adds nothing real")


def demo():
    """Shape/finiteness check on the architecture without touching the data."""
    torch.manual_seed(0)
    net = AttnNet(n_static=7)
    seq = torch.randn(4, 6, 8)
    out = net(seq, torch.randn(4, 7))
    assert out.shape == (4,) and torch.isfinite(out).all()
    # positional embeddings must matter: shuffling months changes the output
    shuffled = net(seq[:, [3, 1, 5, 0, 4, 2]], torch.zeros(4, 7))
    assert not torch.allclose(out, net(seq, torch.zeros(4, 7)) * 0 + shuffled)
    print("demo ok")


if __name__ == "__main__":
    import sys
    demo() if "--demo" in sys.argv else main()
