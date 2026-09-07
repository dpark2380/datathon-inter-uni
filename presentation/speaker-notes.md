# Speaker Notes: Stream 1, Credit Card Default

Matches the notes saved on each slide in `finalist-presentation.pptx`. Slide titles
here are the plain on-slide titles (not sentence-style takeaways); the takeaway for
each slide is spoken, not printed as the heading. Total spoken text lands at
approximately four minutes across the eight slides; the appendix slides carry no
timed notes and are presented only if a judge asks.

| # | Title | Budget | Seconds |
|---|---|---|---|
| 1 | Overview | 0:00-0:20 | 20s |
| 2 | Data Cleaning | 0:20-0:50 | 30s |
| 3 | Feature Engineering | 0:50-1:30 | 40s |
| 4 | Experimentation | 1:30-2:15 | 45s |
| 5 | The Models | 2:15-2:38 | 23s |
| 6 | Validation & Results | 2:38-2:56 | 18s |
| 7 | Model Insights & Limitations | 2:56-3:41 | 45s |
| 8 | Real-World Application | 3:41-4:01 | 20s |

---

**1. Overview (0:00, 20s)**

> We're predicting each customer's probability of default next month, scored by binary log loss, so calibration matters as much as ranking. Twenty-four thousand labelled customers, six thousand held out, twenty-three raw columns, a 22.12 percent default rate. Our final hidden-test log loss is 0.40982, from a blend of 0.45 LightGBM, 0.30 GRU, and 0.25 TabPFN.

**2. Data Cleaning (0:20, 30s)**

> The data had no missing values and no malformed rows. Cleaning stayed deliberately narrow. EDUCATION codes 0, 5, and 6 got grouped into "other", 290 rows; MARRIAGE code 0 grouped into "other", 42 rows. We kept PAY status values of minus 1 and minus 2 as features, since they mean paid-in-full and inactive account, not missing data. No outlier removal, since large balances can be genuine signal. No resampling, since changing the 22 percent base rate would hurt calibration. Twenty-three raw columns became 81 engineered features.

**3. Feature Engineering (0:50, 40s)**

> The data gives statement balances and payments, but never states what a customer actually spent. So we worked it out ourselves. New activity equals current bill minus previous bill plus payment. That matters because a rising balance from new spending is a different risk story than a rising balance from missed payments. This one feature family improved out-of-fold log loss by 0.00046 and hidden-test log loss by 0.00173, the largest gain of the project. Six other feature families we tried never cleared our 0.0005 noise threshold.

**4. Experimentation (1:30, 45s)**

> We ran 41 experiments across 10 model families, 7 feature families, and 135 LightGBM configurations. Only a handful of decisions actually moved the needle. We switched from AUC to log loss, added a GRU to preserve month-to-month order, added the spending decomposition, the biggest single jump, then added TabPFN. Score fell from 0.41260 to 0.40982. Plenty of experiments went nowhere. 135 LightGBM configs found nothing reliable, a 1D-CNN just duplicated the GRU's signal, a survival model improved out-of-fold but got worse on hidden test so we dropped it, and isotonic calibration looked great until nested cross-validation revealed it was leakage. Disagreement between models only helps when it carries real signal.

**5. The Models (2:15, 23s)**

> Three models, three different views of the same customer. LightGBM carries the most weight and reads 81 engineered features as one aggregate profile. The GRU reads six months as an ordered sequence, telling recovery apart from deterioration. TabPFN applies a pretrained prior, with zero parameters fit to our data.

**6. Validation & Results (2:38, 18s)**

> All three share identical cross-validation folds, so a 0.0005 gap is just noise. The blend reaches 0.42136 out-of-fold and 0.40982 on hidden test, a plain weighted average, no calibration, just a clip to the probability bounds.

**7. Model Insights & Limitations (2:56, 45s)**

> The strongest predictors are all about recent, severe delinquency, low available credit, and weak payment coverage. This SHAP view explains the LightGBM component specifically, not the full ensemble. But the model has real blind spots. Our twenty worst-predicted customers all defaulted despite scoring just 2 to 4 percent risk, every one with a clean payment history, meaning the real driver isn't in this data at all. And the EDUCATION "other" subgroup, 387 customers, 28 defaults, gets over-predicted by about 61 percent, with an AUC of just 0.645 against roughly 0.79 elsewhere. That subgroup shouldn't drive automated decisions.

**8. Real-World Application (3:41, 20s)**

> These probabilities should support manual review, early customer assistance, limit monitoring, and portfolio prioritization, not function as an automatic approve or decline rule. A false positive means a reliable customer faces unnecessary friction; a false negative means a missed default and a missed chance to help early. Thresholds should come from business cost and review capacity, not the competition score. Some defaults aren't predictable from this data, one subgroup is unreliable, and TabPFN adds licensing and compute cost for a small gain. Understanding the account mechanics and validating carefully mattered more than adding model complexity.

---

### Delivery notes

- Numbers to know cold without looking at a slide: **0.40982** (final), **0.00173** (spend-decomposition gain), **0.645** (the "other" AUC).
- Everything cut from the eight main slides (full hyperparameters, all 41 experiments, the weight-search surface, calibration comparison, SHAP waterfalls, the full fairness table, reproducibility pipeline, disclosures) lives in the appendix slides (A1-A7) at the end of the deck. Jump to the relevant one directly if a judge asks rather than paging through.
- See `presentation/qa-prep.md` for prepared answers to the judging-panel questions listed in `slide_plan.md`, each tagged with the appendix slide that backs it up.
- This file mirrors `presentation/script.md` in content. `script.md` is the presenter's own rehearsal copy with its own section headers; this file is the deliverable synced word-for-word to the notes saved in `finalist-presentation.pptx`.
