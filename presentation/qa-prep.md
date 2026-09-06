# Q&A Prep — Stream 1: Credit Card Default

3 minutes, unscripted. Answers below are short (say the headline number first, then one sentence of "why"), with the source cited so you can go find the exact passage if a judge pushes further.

## Stream-specific questions (from the brief)

**Q: What were the strongest indicators of default risk?**
Recency of lateness dominates over raw balances. Top SHAP drivers: `months_since_late`, `PAY_0` (most recent repayment status), and `recent_late`. Two of our engineered spending-decomposition features also rank in the top 15. Concretely: a customer with `PAY_0`/`recent_late`/`months_since_late` all pointing bad reaches 84.9% predicted risk; a customer with six clean months and high available credit reaches 2.2%.
*Source: `models/interpret.py` → `output/analysis/shap_summary.png`; REPORT.md §1.3, §1.7.*

**Q: Where did your model make the most mistakes?**
Two distinct failure modes, not one. (1) A subgroup failure: `EDUCATION="other"` (n=387, an artifact of our own cleaning step collapsing undocumented codes) is over-predicted 61% relative — actual default rate 7.24% vs. mean predicted 11.62% — and its AUC drops to 0.645 against ~0.79 for every other education group. (2) A data-ceiling failure: the 20 largest-residual customers were all real defaulters the model scored at only 2–4% risk, and every one had a clean payment history, moderate utilization, and a healthy limit — nothing in the given columns flags them.
*Source: `output/analysis/fairness_audit.txt`; REPORT.md §1.7 Limitations.*

**Q: What are the consequences of false positives vs. false negatives here?**
A false negative extends credit to someone who then defaults — a direct, quantifiable dollar loss on that account. A false positive declines or restricts a customer who would have paid — a smaller, harder-to-see cost in foregone revenue, but a real fairness cost too, because our errors are *not evenly distributed*: the "other" education segment is specifically over-predicted, so a hard cutoff would disproportionately generate false positives there. That's the direct argument for not collapsing this to a single threshold.
*Source: fairness_audit.txt calibration-ratio findings; REPORT.md §1.2 (cleaning provenance of the "other" group).*

**Q: How could your model's predictions support real credit-risk decisions? What's the plan?**
A three-tier policy, not a single cutoff: auto-approve the low-risk majority, auto-decline only the extreme high-risk tail, and route the middle band — plus any customer in a subgroup we know is miscalibrated (like "other" education) — to human underwriting review. Practically: (1) recompute the fairness audit whenever the model is retrained, since the failure mode is tied to a specific cleaning decision and could resurface or shift; (2) monitor calibration drift over time, since the whole approach assumes the 22.121% base rate holds; (3) treat probability of default as an input to pricing/limit decisions, not just approve/decline — e.g., lower limits or closer monitoring for the review band rather than an outright decline.

## Generic methodology questions likely to come up

**Q: Why log loss instead of AUC or accuracy?**
Log loss was the competition's actual scoring metric, and it rewards *calibration* (how close the predicted probability is to the true rate), not just ranking. We caught this early and switched model selection and blend weighting from AUC to log loss mid-project — the two metrics don't always agree (LightGBM's incumbent hyperparameters were originally tuned against AUC and were never beaten on log loss by 135 later configs).
*Source: REPORT.md §1.1, §1.5.*

**Q: Why blend three models instead of just shipping the best one?**
Because the three differ in *how they read a customer*, not just in algorithm — LightGBM reads flat aggregate statistics, the GRU reads the same six months as an ordered trajectory, TabPFN applies a pretrained prior via in-context learning. Seven other model families that were "just a different algorithm over the same aggregate view" (CatBoost, ExtraTrees, logistic regression, MLP, autoencoder, 1D-CNN, multi-task GRU) all got weight 0.00 in the final search — disagreement alone isn't enough, the disagreement has to be informative. That said, be honest if pushed: LightGBM alone reaches 0.42211 OOF, the full blend reaches 0.42136 — the other two components buy about 0.0007 for 180 extra model fits.
*Source: REPORT.md §1.1, §1.5, §1.7 Limitations.*

**Q: Why no resampling or class weighting for the 22% imbalance?**
Because 22.121% is the actual rate the test set is drawn from, and log loss rewards predicting that rate honestly — rebalancing would shift predicted probabilities away from the true base rate. This is also why TabPFN was run with `balance_probabilities=False`.
*Source: REPORT.md §1.2.*

**Q: Did you calibrate the model?**
No, and that's a finding, not an oversight. Four calibration schemes were tested under nested cross-validation — Platt scaling, per-segment calibration, isotonic regression, and a base-rate shrinkage — and all of them made log loss worse. Isotonic regression is the most interesting result: it initially looked like a *large* win because it was fitted and scored on the same out-of-fold rows (a leakage bug); re-run correctly under nested CV, it became the single worst result in the entire experiment table (−0.00330). We caught and fixed that mid-project, and every fitted step afterward used nested CV.
*Source: REPORT.md §1.6, §1.7 Observation 3.*

**Q: Why use TabPFN given the reproducibility cost you flagged?**
It earned its 25% weight on measured OOF/hidden-test performance (0.42345 OOF, matching the tuned GRU with zero tuning), and its errors are different enough from the tree/sequence models to be worth keeping. We disclosed the cost explicitly rather than hiding it: it requires a Prior Labs licence and API token, and takes ~3h10m on CPU — which is why we committed the prediction vectors in `artifacts/` so the submission is reconstructible in seconds without rerunning it.
*Source: REPORT.md §1.5, §1.7 Limitations, Required Disclosure.*

**Q: What would you do with more time?**
Not more tuning — 135 LightGBM configs, 10 model families, 8 ensembling/calibration schemes, and 5 techniques borrowed from a much larger AMEX Kaggle competition all failed to move the needle further. The pattern across 41 experiments is that new *information* wins and new *algorithms* don't. The honest next step is a genuinely new signal about the data, not a variant of what's already been tried. Two concrete leads: investigate whether the "other"-education miscalibration can be fixed by not collapsing those codes together (they may not actually be a homogeneous group), and revisit the TabPFN device setting — it was run on CPU due to an untested MPS-compatibility assumption that turned out to be wrong, and the recovered time could have funded one more real experiment before the deadline.
*Source: REPORT.md §1.7 (Observation 4, Limitations, "honest position on the ceiling").*
