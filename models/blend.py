"""Blend the LightGBM, GRU-sequence and TabPFN predictions into the submission.

Run lgbm_model.py, seq_model.py and tabpfn_model.py first. This loads their
saved OOF/test vectors, searches the weight simplex for the mix that
minimises OOF log loss (the competition metric), checks whether Platt or
isotonic recalibration helps further, and writes the submitted file:
.output/predictions_blend.csv.

Two details needed to reproduce the submitted result:

  * LightGBM's component is test_lgbm_full.npy (the pure full-data refit),
    not the 50/50 fold/refit mix in test_lgbm.npy -- the pure refit scored
    0.41032 vs 0.41038 when tested in isolation.
  * rejected/nn_model.py's MLP is deliberately excluded: the weight search
    always drove it to 0.00 once the GRU existed, so it's kept only as a
    tested-and-rejected model, not carried here.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from common import ID, OUT, TARGET, clean, folds, score

MODELS = ("lgbm", "seq", "tabpfn")
GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)
EPS = 1e-4


def main():
    y = clean(pd.read_csv("datasets/train_clean.csv", dtype={ID: str}))[TARGET].to_numpy()
    try:
        oof = {m: np.load(OUT / f"oof_{m}.npy") for m in MODELS}
        test = {m: np.load(OUT / f"test_{m}.npy") for m in MODELS}
        # see the module docstring: the submission uses the pure full-data refit
        test["lgbm"] = np.load(OUT / "test_lgbm_full.npy")
    except FileNotFoundError as e:
        raise SystemExit(
            f"missing {e.filename} -- run lgbm_model.py, seq_model.py and tabpfn_model.py first"
        )

    for m in MODELS:
        score(m, y, oof[m])

    print("\ncorrelation with lgbm:  " + "  ".join(
        f"{m} {np.corrcoef(oof['lgbm'], oof[m])[0, 1]:.4f}" for m in MODELS if m != "lgbm"
    ))

    # search the weight simplex over all three models
    print("\nweight search (by OOF log loss, the competition metric):")
    losses = {}
    for w_seq in GRID:
        for w_tab in GRID:
            if w_seq + w_tab > 1:
                continue
            w = (round(1 - w_seq - w_tab, 2), round(w_seq, 2), round(w_tab, 2))
            losses[w] = log_loss(y, sum(wi * oof[m] for wi, m in zip(w, MODELS)))

    best = min(losses, key=losses.get)
    solo = losses[(1.0, 0.0, 0.0)]
    gain = solo - losses[best]
    for w in sorted(losses, key=losses.get)[:5]:
        print(f"  (lgbm,seq,tabpfn)={w}  log loss {losses[w]:.5f}")
    print(f"\nbest weights (lgbm,seq,tabpfn)={best}  log loss {losses[best]:.5f}"
          f"  (-{gain:.5f} vs lgbm alone)")
    if gain < 0.0005:
        print("gain is under 0.0005, i.e. inside fold noise -- plain lgbm is the safer pick")

    oof_blend = sum(wi * oof[m] for wi, m in zip(best, MODELS))
    test_blend = sum(wi * test[m] for wi, m in zip(best, MODELS))

    # Check whether recalibrating the blend lowers OOF log loss further.
    # Isotonic regression can overfit the exact points it's scored on, so
    # this uses its own nested CV (common.folds) instead of fit-and-score on
    # the same rows -- otherwise a flexible calibrator always "wins". "none"
    # stays in the running, so calibration can only help, never hurt.
    def logit(p):
        return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))).reshape(-1, 1)

    def cv_oof(fit_predict):
        out = np.zeros_like(oof_blend)
        for tr, va in folds(oof_blend.reshape(-1, 1), y):
            out[va] = fit_predict(oof_blend[tr], y[tr], oof_blend[va])
        return out

    def platt_fit_predict(p_tr, y_tr, p_va):
        m = LogisticRegression(C=1e10).fit(logit(p_tr), y_tr)
        return m.predict_proba(logit(p_va))[:, 1]

    def iso_fit_predict(p_tr, y_tr, p_va):
        m = IsotonicRegression(out_of_bounds="clip").fit(p_tr, y_tr)
        return m.predict(p_va)

    print("\ncalibration check on the blend (cross-validated OOF log loss):")
    candidates = {
        "none": losses[best],
        "platt": log_loss(y, cv_oof(platt_fit_predict)),
        "isotonic": log_loss(y, cv_oof(iso_fit_predict)),
    }
    for name, ll in candidates.items():
        print(f"  {name:<9} log loss {ll:.5f}")

    best_cal = min(candidates, key=candidates.get)
    print(f"\nbest calibration: {best_cal}")

    if best_cal == "none":
        preds = test_blend
    elif best_cal == "platt":
        m = LogisticRegression(C=1e10).fit(logit(oof_blend), y)
        preds = m.predict_proba(logit(test_blend))[:, 1]
    else:
        m = IsotonicRegression(out_of_bounds="clip").fit(oof_blend, y)
        preds = m.predict(test_blend)
    preds = np.clip(preds, EPS, 1 - EPS)

    ids = pd.read_csv("datasets/test.csv", usecols=[ID], dtype={ID: str})[ID]
    out = pd.DataFrame({ID: ids, "prob_default": preds})
    assert len(out) == len(ids) and out["prob_default"].between(0, 1).all()
    out.to_csv(OUT / "predictions_blend.csv", index=False)
    print(f"wrote {OUT / 'predictions_blend.csv'}  mean predicted {preds.mean():.4f}")


if __name__ == "__main__":
    main()
