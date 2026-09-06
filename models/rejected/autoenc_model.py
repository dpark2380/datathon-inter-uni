"""Bagged supervised autoencoder classifier (Abdoli et al. 2021), plus an
optional semi-supervised use of the unlabeled test features.

The encoder's latent space is shaped by BOTH losses at once:

    loss = BCE(classifier(z), y) + ALPHA * MSE(decoder(z), x)

so the representation has to stay reconstructable (denoised, keeps the
structure of the whole feature space) while still being linearly separable for
the default head. Input is corrupted with gaussian noise, the reconstruction
target is the clean input -- a denoising AE, which is where the regularisation
comes from.

Semi-supervised bit: the 6,000 unlabeled test rows carry no target, but they do
carry features, so they can contribute to the *reconstruction* term. That is
transductive but not leakage -- no label of any kind is involved. Toggle with
AE_UNLABELED=0/1.

Bagging: several seeds per fold, averaged, as in the paper (they bootstrap; we
average seeds, which is what every other model in this repo does and what
actually moved the needle in nn_model.py).

RESULT (null): solo OOF 0.42685 / auc 0.78516 over 6 repeats x 3 seeds, corr
0.977 with lgbm and 0.987 with seq, and a weight grid-search over
(lgbm, seq, ae) puts it at weight 0.000 -- the incumbent 0.42155 blend is
unchanged. A 1-repeat sweep showed the reconstruction term is what hurts:
ALPHA 0 -> 0.4257, 0.3 -> 0.4265, 1 -> 0.4275, 3 -> 0.4304, i.e. the more the
latent has to reconstruct, the worse it classifies, and ALPHA=0 is just the
plain MLP that was already rejected. Adding the unlabeled test rows to the
reconstruction term also hurt slightly (0.42750 vs 0.42670 without). Kept for
the record; not worth putting in the blend.

Env knobs for quick pilots: AE_REPEATS (default all of common.REPEATS),
AE_SEEDS (default 3), AE_UNLABELED (default 1), AE_NAME (output name).
"""

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
from pathlib import Path

# This script lives one level below the pipeline modules it imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import CATS, REPEATS, folds, load, save, score

LATENT = int(os.environ.get("AE_LATENT", 32))
HIDDEN = 128
ALPHA = float(os.environ.get("AE_ALPHA", 1.0))   # weight on the reconstruction loss
NOISE = float(os.environ.get("AE_NOISE", 0.2))   # denoising corruption std (inputs are standardized)
DROPOUT = 0.3
MAX_EPOCHS = 200
PATIENCE = 15
BATCH = 512
LR = 2e-3
CLIP = 5.0         # standardized features clipped to tame the fat tails
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

N_SEEDS = int(os.environ.get("AE_SEEDS", 3))
USE_UNLABELED = os.environ.get("AE_UNLABELED", "1") == "1"
REPS = tuple(int(r) for r in os.environ["AE_REPEATS"].split(",")) if os.environ.get("AE_REPEATS") else REPEATS
NAME = os.environ.get("AE_NAME", "ae")


class SAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, LATENT), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(LATENT, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, d),
        )
        self.clf = nn.Sequential(
            nn.Dropout(DROPOUT), nn.Linear(LATENT, 32), nn.ReLU(),
            nn.Dropout(0.15), nn.Linear(32, 1),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.clf(z).squeeze(1), self.dec(z)


def fit_predict(x_tr, y_tr, x_va, y_va, x_te, x_unlab, seed):
    """One bagged member: train with early stopping on validation log loss."""
    torch.manual_seed(seed)
    model = SAE(x_tr.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()

    t = lambda a: torch.tensor(a, device=DEV)
    xt, yt = t(x_tr), t(y_tr.astype(np.float32))
    xv, xe = t(x_va), t(x_te)
    xu = t(x_unlab) if x_unlab is not None else None

    best, best_state, bad, n = np.inf, None, 0, len(x_tr)
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            xb = xt[perm[i : i + BATCH]]
            yb = yt[perm[i : i + BATCH]]
            opt.zero_grad()
            logit, rec = model(xb + NOISE * torch.randn_like(xb))
            loss = bce(logit, yb) + ALPHA * mse(rec, xb)
            if xu is not None:
                ub = xu[torch.randint(len(xu), (len(xb),), device=DEV)]
                _, urec = model(ub + NOISE * torch.randn_like(ub))
                loss = loss + ALPHA * mse(urec, ub)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(xv)[0]).cpu().numpy()
        ll = log_loss(y_va, np.clip(p, 1e-7, 1 - 1e-7))
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
        return (torch.sigmoid(model(xv)[0]).cpu().numpy(),
                torch.sigmoid(model(xe)[0]).cpu().numpy(), best)


def main():
    X, y, X_test, ids = load()
    num = [c for c in X.columns if c not in CATS]
    for df in (X, X_test):
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    n_models = 5 * N_SEEDS * len(REPS)
    print(f"seeds={N_SEEDS} repeats={REPS} unlabeled={USE_UNLABELED} dev={DEV}", flush=True)

    for rep in REPS:
        for tr, va in folds(X, y, seed=rep):
            # fit preprocessing on the training fold only
            imp = SimpleImputer(strategy="median").fit(X.iloc[tr][num])
            sc = StandardScaler().fit(imp.transform(X.iloc[tr][num]))
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X.iloc[tr][CATS])

            def prep(df):
                z = np.clip(sc.transform(imp.transform(df[num])), -CLIP, CLIP)
                return np.hstack([z, ohe.transform(df[CATS])]).astype(np.float32)

            x_tr, x_va, x_te = prep(X.iloc[tr]), prep(X.iloc[va]), prep(X_test)
            x_unlab = x_te if USE_UNLABELED else None

            for seed in range(N_SEEDS):
                p_va, p_te, _ = fit_predict(x_tr, y[tr], x_va, y[va], x_te,
                                            x_unlab, seed + 100 * rep)
                oof[va] += p_va / (N_SEEDS * len(REPS))
                test_pred += p_te / n_models
        print(f"repeat {rep} done  running auc "
              f"{roc_auc_score(y, oof * len(REPS) / (REPS.index(rep) + 1)):.5f}", flush=True)

    oof = np.clip(oof, 1e-7, 1 - 1e-7)
    test_pred = np.clip(test_pred, 1e-7, 1 - 1e-7)
    score(f"{NAME} OOF", y, oof)
    save(NAME, oof, test_pred, ids)

    # does it earn any weight next to the incumbent lgbm + seq blend?
    ref = {m: np.load(f".output/oof_{m}.npy") for m in ("lgbm", "seq")}
    for m, o in ref.items():
        print(f"corr with {m} OOF {np.corrcoef(oof, o)[0, 1]:.4f}")
    grid = np.round(np.arange(0, 1.001, 0.025), 3)
    ll = lambda a, b: log_loss(y, a * ref["lgbm"] + b * ref["seq"] + (1 - a - b) * oof)
    cand = [(ll(a, b), a, b) for a in grid for b in grid if a + b <= 1]
    best = min(cand)
    pair = min(c for c in cand if abs(c[1] + c[2] - 1) < 1e-9)
    print(f"\nlgbm+seq only : {pair[0]:.5f}  w=({pair[1]}, {pair[2]})")
    print(f"with {NAME:<9}: {best[0]:.5f}  w=(lgbm {best[1]}, seq {best[2]}, {NAME} {round(1 - best[1] - best[2], 3)})")
    print(f"gain {pair[0] - best[0]:.5f}"
          + ("  -- under 0.0005, inside fold noise" if pair[0] - best[0] < 5e-4 else ""))


if __name__ == "__main__":
    main()
