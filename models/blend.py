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
from sklearn.metrics import log_loss

from common import ID, OUT, TARGET, clean, folds, score

MODELS = ("lgbm", "seq", "tabpfn")
GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)
EPS = 1e-4


def load_predictions():
    """Load the aligned OOF and test vectors produced by the three models."""
    try:
        oof = {model: np.load(OUT / f"oof_{model}.npy") for model in MODELS}
        test = {model: np.load(OUT / f"test_{model}.npy") for model in MODELS}
        # see the module docstring: the submission uses the pure full-data refit
        test["lgbm"] = np.load(OUT / "test_lgbm_full.npy")
    except FileNotFoundError as e:
        raise SystemExit(
            f"missing {e.filename} -- run lgbm_model.py, seq_model.py and tabpfn_model.py first"
        )
    return oof, test


def search_weights(y, oof):
    """Return every grid-search loss and the lowest-loss weight tuple."""
    losses = {}
    for w_seq in GRID:
        for w_tab in GRID:
            if w_seq + w_tab > 1:
                continue
            weights = (
                round(1 - w_seq - w_tab, 2),
                round(w_seq, 2),
                round(w_tab, 2),
            )
            blend = sum(weight * oof[model] for weight, model in zip(weights, MODELS))
            losses[weights] = log_loss(y, blend)
    return losses, min(losses, key=losses.get)


def blend_predictions(predictions, weights):
    """Combine aligned model probabilities in MODELS order."""
    return sum(weight * predictions[model] for weight, model in zip(weights, MODELS))


def logit(probabilities):
    clipped = np.clip(probabilities, EPS, 1 - EPS)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def cross_validated_calibration(y, probabilities, fit_predict):
    """Generate leakage-free OOF predictions for one calibration method."""
    calibrated = np.zeros_like(probabilities)
    for train_rows, valid_rows in folds(probabilities.reshape(-1, 1), y):
        calibrated[valid_rows] = fit_predict(
            probabilities[train_rows], y[train_rows], probabilities[valid_rows]
        )
    return calibrated


def platt_predict(train_probabilities, train_target, valid_probabilities):
    model = LogisticRegression(C=1e10).fit(logit(train_probabilities), train_target)
    return model.predict_proba(logit(valid_probabilities))[:, 1]


def isotonic_predict(train_probabilities, train_target, valid_probabilities):
    model = IsotonicRegression(out_of_bounds="clip").fit(
        train_probabilities, train_target
    )
    return model.predict(valid_probabilities)


def calibration_scores(y, oof_blend, uncalibrated_loss):
    """Compare no calibration with nested-CV Platt and isotonic calibration."""
    return {
        "none": uncalibrated_loss,
        "platt": log_loss(
            y, cross_validated_calibration(y, oof_blend, platt_predict)
        ),
        "isotonic": log_loss(
            y, cross_validated_calibration(y, oof_blend, isotonic_predict)
        ),
    }


def calibrate(calibration, y, oof_blend, test_blend):
    """Fit the selected calibrator on all OOF rows and apply it to test rows."""
    if calibration == "none":
        return test_blend
    if calibration == "platt":
        model = LogisticRegression(C=1e10).fit(logit(oof_blend), y)
        return model.predict_proba(logit(test_blend))[:, 1]
    model = IsotonicRegression(out_of_bounds="clip").fit(oof_blend, y)
    return model.predict(test_blend)


def save_submission(predictions):
    ids = pd.read_csv("datasets/test.csv", usecols=[ID], dtype={ID: str})[ID]
    output = pd.DataFrame({ID: ids, "prob_default": predictions})
    assert len(output) == len(ids) and output["prob_default"].between(0, 1).all()
    output.to_csv(OUT / "predictions_blend.csv", index=False)


def main():
    y = clean(pd.read_csv("datasets/train_clean.csv", dtype={ID: str}))[TARGET].to_numpy()
    oof, test = load_predictions()

    for model in MODELS:
        score(model, y, oof[model])

    print("\ncorrelation with lgbm:  " + "  ".join(
        f"{model} {np.corrcoef(oof['lgbm'], oof[model])[0, 1]:.4f}"
        for model in MODELS
        if model != "lgbm"
    ))

    # search the weight simplex over all three models
    print("\nweight search (by OOF log loss, the competition metric):")
    losses, best_weights = search_weights(y, oof)
    lightgbm_loss = losses[(1.0, 0.0, 0.0)]
    ensemble_gain = lightgbm_loss - losses[best_weights]
    for weights in sorted(losses, key=losses.get)[:5]:
        print(f"  (lgbm,seq,tabpfn)={weights}  log loss {losses[weights]:.5f}")
    print(
        f"\nbest weights (lgbm,seq,tabpfn)={best_weights}  "
        f"log loss {losses[best_weights]:.5f}  "
        f"(-{ensemble_gain:.5f} vs lgbm alone)"
    )
    if ensemble_gain < 0.0005:
        print("gain is under 0.0005, i.e. inside fold noise -- plain lgbm is the safer pick")

    oof_blend = blend_predictions(oof, best_weights)
    test_blend = blend_predictions(test, best_weights)

    # Check whether recalibrating the blend lowers OOF log loss further.
    # Isotonic regression can overfit the exact points it's scored on, so
    # this uses its own nested CV (common.folds) instead of fit-and-score on
    # the same rows -- otherwise a flexible calibrator always "wins". "none"
    # stays in the running, so calibration can only help, never hurt.
    print("\ncalibration check on the blend (cross-validated OOF log loss):")
    candidates = calibration_scores(y, oof_blend, losses[best_weights])
    for name, loss in candidates.items():
        print(f"  {name:<9} log loss {loss:.5f}")

    selected_calibration = min(candidates, key=candidates.get)
    print(f"\nbest calibration: {selected_calibration}")

    preds = np.clip(
        calibrate(selected_calibration, y, oof_blend, test_blend), EPS, 1 - EPS
    )
    save_submission(preds)
    print(f"wrote {OUT / 'predictions_blend.csv'}  mean predicted {preds.mean():.4f}")


if __name__ == "__main__":
    main()
