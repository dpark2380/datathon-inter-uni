"""Interpretability + calibration diagnostics for the shipped blend --
reporting only, no effect on the submission.

Run after lgbm_model.py, seq_model.py and tabpfn_model.py. Writes two PNGs
to output/analysis/:

1. SHAP plots on a LightGBM fold model: which features drive each
   prediction, globally (summary plot) and for two individual customers
   (waterfall plots).
2. A reliability diagram for LightGBM, GRU, TabPFN and the final blend's
   OOF predictions: does "30% predicted default" actually default ~30% of
   the time?

Answers the brief's explicit "interpretability" and "calibration" callouts.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.calibration import calibration_curve

from common import OUT, folds, load
from lgbm_model import PARAMS

OUT_DIR = Path("output/analysis")
WEIGHTS = {"lgbm": 0.45, "seq": 0.30, "tabpfn": 0.25}  # must match blend.py


def shap_plots(X, y):
    import lightgbm as lgb

    tr, va = next(folds(X, y))
    m = lgb.LGBMClassifier(**PARAMS, random_state=0)
    m.fit(
        X.iloc[tr],
        y[tr],
        eval_X=X.iloc[va],
        eval_y=y[va],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )

    explainer = shap.TreeExplainer(m)
    Xv = X.iloc[va]
    sv = explainer(Xv)

    plt.figure()
    shap.summary_plot(sv, Xv, max_display=15, show=False)
    plt.title("Global feature impact on predicted default probability")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary.png", dpi=150)
    plt.close()

    proba = m.predict_proba(Xv)[:, 1]
    order = np.argsort(proba)
    for label, idx in [("highest_risk", order[-1]), ("lowest_risk", order[0])]:
        plt.figure()
        shap.plots.waterfall(sv[idx], max_display=12, show=False)
        plt.title(f"{label}: predicted p(default)={proba[idx]:.3f}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"shap_waterfall_{label}.png", dpi=150)
        plt.close()

    print(f"wrote shap_summary.png and 2 waterfall plots to {OUT_DIR}/")


def reliability_diagram(y):
    oof = {m: np.load(OUT / f"oof_{m}.npy") for m in WEIGHTS}
    oof["blend"] = sum(w * oof[m] for m, w in WEIGHTS.items())

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    for name, p in oof.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.xlabel("mean predicted probability (per decile bin)")
    plt.ylabel("observed default rate")
    plt.title("Reliability diagram (OOF predictions, quantile-binned)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "reliability_diagram.png", dpi=150)
    plt.close()
    print(f"wrote reliability_diagram.png to {OUT_DIR}/")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, _, _ = load()
    shap_plots(X, y)
    reliability_diagram(y)


if __name__ == "__main__":
    main()
