"""1D-CNN over the same 6-month panel as seq_model.py.

Different inductive bias from the GRU, which is the whole point of building it:
a convolution fires on a local pattern (a two-month deterioration burst, a
recovery step) *wherever in the window it occurs*, while the GRU compresses the
history into one running state. Parallel kernel widths 2 and 3 plus global
max+mean pooling let the head see "did this pattern ever happen" as well as
"how much of it on average".

Reuses seq_model's panel construction and training loop; only the architecture
differs. Same folds via common.folds so the OOF vector blends with the rest.
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

from common import CATS, ID, TARGET, clean, folds, load, save, score
from seq_model import fit_predict, panel

SEEDS = tuple(range(6))
CHANNELS = 32


class ConvNet(nn.Module):
    def __init__(self, n_static, n_chan=8, ch=CHANNELS):
        super().__init__()
        # two kernel widths in parallel: 2-month and 3-month motifs
        self.k2 = nn.Sequential(nn.Conv1d(n_chan, ch, 2, padding=1), nn.ReLU())
        self.k3 = nn.Sequential(nn.Conv1d(n_chan, ch, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(4 * ch + n_static, 96),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

    def forward(self, seq, static):
        x = seq.transpose(1, 2)  # (N, C, T) for Conv1d
        pooled = []
        for conv in (self.k2, self.k3):
            h = conv(x)
            pooled += [h.max(dim=2).values, h.mean(dim=2)]
        return self.head(torch.cat(pooled + [static], dim=1)).squeeze(1)


def main():
    train_raw = clean(pd.read_csv("datasets/train_clean.csv", dtype={ID: str}))
    test_raw = clean(pd.read_csv("datasets/test.csv", dtype={ID: str}))
    X, y, X_test, ids = load()

    seq_all, seq_test = panel(train_raw), panel(test_raw)
    num = [c for c in X.columns if c not in CATS]
    for df in (X, X_test):
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    splits = list(folds(X, y))
    n_models = len(splits) * len(SEEDS)

    for fold, (tr, va) in enumerate(splits, 1):
        imp = SimpleImputer(strategy="median").fit(X.iloc[tr][num])
        sc = StandardScaler().fit(imp.transform(X.iloc[tr][num]))
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X.iloc[tr][CATS])

        def static(df):
            return np.hstack([sc.transform(imp.transform(df[num])),
                              ohe.transform(df[CATS])]).astype(np.float32)

        st_tr, st_va, st_te = static(X.iloc[tr]), static(X.iloc[va]), static(X_test)

        mu = seq_all[tr].reshape(-1, seq_all.shape[2]).mean(0)
        sd = seq_all[tr].reshape(-1, seq_all.shape[2]).std(0) + 1e-6
        norm = lambda a: ((a - mu) / sd).astype(np.float32)

        for seed in SEEDS:
            p_va, p_te, _ = fit_predict(
                norm(seq_all[tr]), st_tr, y[tr],
                norm(seq_all[va]), st_va, y[va],
                norm(seq_test), st_te, seed,
                make_model=lambda n: ConvNet(n),
            )
            oof[va] += p_va / len(SEEDS)
            test_pred += p_te / n_models
        oof[va] = np.clip(oof[va], 1e-7, 1 - 1e-7)
        print(f"fold {fold}  auc {roc_auc_score(y[va], oof[va]):.5f}  "
              f"log loss {log_loss(y[va], oof[va]):.5f}", flush=True)

    print()
    test_pred = np.clip(test_pred, 1e-7, 1 - 1e-7)
    score("cnn OOF", y, oof)
    save("cnn", oof, test_pred, ids)
    print(f"base rate {y.mean():.4f}  mean predicted {test_pred.mean():.4f}")
    for other in ("lgbm", "seq"):
        print(f"corr with {other}: {np.corrcoef(oof, np.load(f'.output/oof_{other}.npy'))[0, 1]:.4f}")


if __name__ == "__main__":
    main()
