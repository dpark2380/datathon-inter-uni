# Speaker Script — Stream 1: Credit Card Default

Timed to a 4:00 talk (≈130 wpm). Total ≈ 520 words including this budget's slack — read it aloud once with a stopwatch before the real run and trim wherever you naturally run long. Segment order matches `slides.html` 1–9.

| Slide | Budget | Words |
|---|---|---|
| 1. Title | 0:00–0:15 | ~35 |
| 2. Data understanding | 0:15–0:35 | ~48 |
| 3. Spend decomposition | 0:35–1:05 | ~70 |
| 4. Experimentation timeline | 1:05–1:35 | ~70 |
| 5. Model selection & validation | 1:35–2:10 | ~66 |
| 6. What drives risk | 2:10–2:40 | ~70 |
| 7. Where the model fails | 2:40–3:15 | ~78 |
| 8. Model to decision | 3:15–3:45 | ~75 |
| 9. Limitations & close | 3:45–4:00 | ~42 |

---

**1. Title (0:00)**

> We built a model to predict whether a credit-card customer will default next month — twenty-four thousand customers, twenty-three raw columns, scored on log loss. Our final submission landed at 0.40982, fifteen thousandths behind first place.

**2. Data understanding (0:15)**

> The data was clean — zero missing values across all twenty-four thousand rows. But it had undocumented category codes: education and marriage each had values the data dictionary never defined. We found them by inspecting every unique value per column, then collapsed them into "other," identically for train and test.

**3. The spend decomposition (0:35)**

> The single biggest win of the project wasn't a better model — it was recovering a number the data never states: how much a customer actually spent. Statement balances and payments don't tell you that directly, but it falls out of one accounting identity. That one feature dropped our hidden-test log loss by 0.00173, the largest gain we measured all project. Six other engineered feature families failed.

**4. Experimentation timeline (1:05)**

> We ran forty-one logged experiments; only four produced gains that survived a noise floor of half a thousandth. Our score fell from a 0.41260 baseline to 0.40982 across eight submissions. One result deserves an honest mention: isotonic calibration looked like a huge win until we caught a leakage bug — under proper validation it became our single worst result.

**5. Model selection & validation (1:35)**

> Our final model blends three components that each read a customer differently: LightGBM as flat statistics at forty-five percent, a GRU reading six months in order at thirty percent, and TabPFN — a pretrained model fit to zero of our parameters — at twenty-five percent. We validated with repeated stratified five-fold cross-validation, moving from 0.42136 out-of-fold to 0.40982 on the hidden test set.

**6. What drives risk (2:10)**

> Interpreting the model with SHAP, the strongest predictors are all about recency of lateness — months since last late, the most recent repayment status, whether they were late recently — more than raw balances. Our highest-risk customer scored an 85% probability of default; our lowest-risk customer scored just 2%, driven by six clean months and healthy available credit.

**7. Where the model fails (2:40)**

> Two honest failure modes. First, the "other" education group — created by our own cleaning step — is over-predicted by 61% relative to its actual rate, with an AUC of just 0.645 versus around 0.79 everywhere else. Second, our twenty worst-predicted customers were all real defaulters we scored at 2–4% risk — every one had a spotless payment history. That signal simply isn't present in these columns.

**8. From model to decision (3:15)**

> For a real credit decision, a false negative extends credit that becomes bad debt — a direct dollar loss. A false positive declines a good customer: lost revenue, and a fairness cost, especially given that "other" miscalibration. So we'd recommend a three-tier policy: auto-approve low risk, auto-decline only the extreme high-risk tail, and route everyone in between — including that fragile subgroup — to human review.

**9. Limitations & close (3:45)**

> LightGBM alone gets us most of the way there; the full ensemble buys a small gain for real reproducibility cost. The levers we can name are exhausted — a better score needs a new insight about the data, not more tuning.

---

### Delivery notes

- The three numbers judges are most likely to probe live: **0.40982** (final), **0.00173** (spend-decomposition gain), **0.645** (the "other" AUC). Know these cold without looking at a slide.
- If running long, cut from slide 4 (timeline) first — it's the most compressible without losing a judging axis, since slide 9 already restates "41 experiments" implicitly via the limitations framing. Do not cut slide 7 or 8 — those are the stream's explicit judging questions (where does it fail, FP/FN, real-world use).
