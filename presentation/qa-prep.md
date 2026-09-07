# Q&A Prep: Stream 1, Credit Card Default

Three minutes, unscripted. Question list is copied verbatim from `slide_plan.md`'s "Likely judging-panel questions" section. Each answer names which appendix slide (in the Figma deck and the pptx) has the supporting evidence, so you can jump straight there if a judge wants to see the numbers.

**Why optimise log loss rather than AUC?**
Log loss is the competition's actual metric, and it scores calibration (how close a predicted probability is to the true rate), not just ranking. We switched model selection and blend weighting from AUC to log loss mid-project once this became clear.
*Slide 1; REPORT.md §1.1.*

**Why was class resampling or weighting not used?**
22.12% is the true rate the test set is drawn from. Rebalancing would shift predicted probabilities away from that rate, and log loss punishes exactly that kind of miscalibration.
*Appendix A7; REPORT.md §1.2.*

**Why retain GRU and TabPFN when LightGBM provides most of the predictive value?**
They read the customer differently (an ordered 6-month sequence, and a pretrained in-context prior) rather than being different algorithms over the same flat view. Their errors are different enough to be worth 0.0007 in log loss. We disclose the cost plainly: 180 extra model fits and two extra dependencies for that gain.
*Slide 5, Appendix A1, A7; REPORT.md §1.5, §1.7 Limitations.*

**How were the ensemble weights selected, and could they be overfitted?**
Grid search over weight triples on out-of-fold log loss. The surface is a broad plateau, not a sharp peak: every LightGBM weight from 0.50 to 0.70 lands within 0.0001 of the optimum, and shrinking the weights toward equal made the hidden-test score worse rather than better, which is the standard test for weights that latched onto noise.
*Appendix A3; REPORT.md §1.6.*

**Why were identical cross-validation folds used for all models?**
So every model's out-of-fold prediction is for the exact same held-out rows, which lets us blend and compare them directly without re-indexing or introducing leakage between components.
*Slide 6; REPORT.md §1.4.*

**Why did OOF and hidden-test improvements sometimes differ?**
Repeated CV and full-data refits are invisible to OOF scoring by construction: an OOF row is only ever predicted by models that didn't see it, while every hidden-test row is predicted by the average of all of them. The sixth CV partition is the clearest example: it helped OOF by 0.00024 and did nothing on hidden test.
*Appendix A7; REPORT.md §1.7 Observation 2.*

**What went wrong with the initial isotonic-calibration experiment?**
It was fitted and scored on the same out-of-fold rows, so it absorbed noise in those rows rather than learning a real correction. Re-run under proper nested cross-validation, it went from looking like the best result to the single worst one (-0.00330). We fixed the methodology and every fitted step afterward used nested CV.
*Slide 4, Appendix A3; REPORT.md §1.7 Observation 3.*

**Why did the survival model get removed after improving OOF?**
It improved OOF log loss by 0.00027, but that gain did not hold on the hidden test set (0.41044 against 0.41038 without it), so it was dropped rather than trusted on the weaker signal.
*Slide 4, Appendix A2.*

**What information does the GRU retain that aggregate features lose?**
Order. Two customers with identical counts of late months, average payment ratios, and so on can still be a recovering customer and a deteriorating one; the aggregate features can't tell them apart, but reading the six months in sequence can.
*Slide 5.*

**What does TabPFN contribute, and how is its pretrained status disclosed?**
It applies a prior learned from pretraining on millions of synthetic datasets, with zero parameters fit to our data, an in-context prediction rather than a trained model. It matched the tuned GRU's OOF score without any task-specific tuning. Disclosed plainly: Prior Labs licence, API token, in-context learning, no parameters fit to this dataset.
*Appendix A1, A6.*

**How can the final submission be reproduced without retraining TabPFN?**
Saved OOF and test prediction vectors are committed under `artifacts/`. Applying the 0.45/0.30/0.25 weighted average and the clip to those vectors reconstructs the submitted file to 9.7e-17, in seconds, without rerunning the roughly 3-hour TabPFN pass.
*Appendix A6.*

**What caused the largest false-negative errors?**
The 20 worst-predicted customers were all real defaulters scored at only 2-4% risk, every one with a clean payment history, moderate utilisation, and a healthy credit limit. Whatever drove those defaults isn't observable in repayment status, balances, payments, or demographics, so no model over these columns can recover it.
*Slide 7; REPORT.md §1.7 Limitations.*

**How should the EDUCATION="other" subgroup issue affect real-world use?**
That subgroup (n=387, 28 defaults) is over-predicted about 61% relative, with an AUC of just 0.645 against roughly 0.79 elsewhere, the only segment where the model does worse than its own base-rate guess. It shouldn't support an automated adverse decision; it should route to manual review instead.
*Slide 7, Appendix A5.*

**What are the relative consequences of false positives and false negatives?**
A false positive puts a reliable customer through unnecessary friction: extra review, reduced credit, or an adverse decision they didn't deserve. A false negative means a missed default, an underestimated loss, and a missed opportunity for early customer support.
*Slide 8.*

**How would an operational decision threshold be selected?**
From business cost, regulatory requirements, and review capacity, not from the competition's log loss score alone. The three-tier continuum on slide 8 (standard process / manual review / verification-support) is a starting shape for that decision, not a fixed threshold.
*Slide 8.*

**What would be investigated with additional time or data?**
Not more tuning: 135 LightGBM configurations, 10 model families, 8 ensembling and calibration schemes, and 5 techniques borrowed from a much larger AMEX Kaggle competition all failed to move the needle further. The next real step is new information about the data, not a variant of what's already been tried, for example, whether the "other" education codes are actually a homogeneous group or should be split.
*Appendix A2; REPORT.md §1.7 Observation 4, "honest position on the ceiling".*
