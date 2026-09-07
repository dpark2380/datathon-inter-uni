"""Regenerate the reliability diagram for the final blend, correctly labeled.

output/analysis/reliability_diagram.png is stale (legend shows "lgbm"/"nn",
not the actual LightGBM + GRU + TabPFN blend). This rebuilds it from the
committed OOF vectors, in the presentation's light palette.

Run: uv run --with matplotlib python3 presentation/scripts/regenerate_reliability_diagram.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
OUT = os.path.join(HERE, "..", "assets")
os.makedirs(OUT, exist_ok=True)

BLUE = "#0071E3"
TEXT = "#1D1D1F"
SECOND = "#6E6E73"

lgbm = np.load(os.path.join(ROOT, "artifacts", "oof_lgbm.npy"))
seq = np.load(os.path.join(ROOT, "artifacts", "oof_seq.npy"))
tabpfn = np.load(os.path.join(ROOT, "artifacts", "tabpfn_6rep", "oof_tabpfn_6rep.npy"))
y = pd.read_csv(os.path.join(ROOT, "datasets", "train_clean.csv"))["default"].to_numpy()

assert len(lgbm) == len(seq) == len(tabpfn) == len(y) == 24000, "row count mismatch"

blend = 0.45 * lgbm + 0.30 * seq + 0.25 * tabpfn


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if mask.sum() == 0:
            continue
        total += mask.sum() * abs(p[mask].mean() - y[mask].mean())
    return float(total / len(p))


metrics = {"lgbm": lgbm, "seq": seq, "tabpfn": tabpfn, "blend": blend}
print("model    brier      ece")
for name, p in metrics.items():
    print(f"{name:<8} {brier(p, y):.5f}   {ece(p, y):.5f}")

# Quantile-binned reliability curve for the blend only (the deck's claim is about the blend)
order = np.argsort(blend)
p_sorted, y_sorted = blend[order], y[order]
bins = 10
bin_edges = np.array_split(np.arange(len(p_sorted)), bins)
mean_pred = [p_sorted[idx].mean() for idx in bin_edges]
mean_obs = [y_sorted[idx].mean() for idx in bin_edges]

fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.plot([0, 1], [0, 1], linestyle="--", color=SECOND, linewidth=1.2, label="perfectly calibrated")
ax.plot(mean_pred, mean_obs, marker="o", color=BLUE, linewidth=2.2, markersize=7, label="final blend (OOF)")
ax.set_xlabel("mean predicted probability (per decile bin)", fontsize=11, color=TEXT)
ax.set_ylabel("observed default rate", fontsize=11, color=TEXT)
ax.set_title("Reliability diagram, final blend, OOF predictions", fontsize=13, color=TEXT)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "reliability_diagram_final.png"), dpi=170, bbox_inches="tight")
print("Saved", os.path.join(OUT, "reliability_diagram_final.png"))
