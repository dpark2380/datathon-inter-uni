# Speaker Script — Stream 1: Credit Card Default

Timed to a 4:00 talk (≈130 wpm). Total ≈ 514 words → ~3:57. Read it aloud once with a stopwatch before the real run and trim wherever you naturally run long. Segment order matches `slides.html` 1–10.

| Slide | Budget | Words |
|---|---|---|
| 1. Title | 0:00–0:15 | ~35 |
| 2. Data understanding | 0:15–0:30 | ~38 |
| 3. Spend decomposition | 0:30–0:55 | ~53 |
| 4. Experimentation timeline | 0:55–1:15 | ~46 |
| 5. How the model actually predicts | 1:15–1:55 | ~72 |
| 6. Model selection & validation | 1:55–2:20 | ~50 |
| 7. What drives risk | 2:20–2:45 | ~58 |
| 8. Where the model fails | 2:45–3:15 | ~64 |
| 9. Model to decision | 3:15–3:40 | ~56 |
| 10. Limitations & close | 3:40–4:00 | ~42 |

---

**1. Title (0:00)**

> We built a model to predict whether a credit-card customer will default next month — twenty-four thousand customers, twenty-three raw columns, scored on log loss. Our final submission landed at 0.40982, fifteen thousandths behind first place.

**2. Data understanding (0:15)**

> The data was clean — zero missing values across twenty-four thousand rows. But education and marriage had undocumented codes, found by inspecting every unique value per column and collapsed into "other," identically for train and test.

**3. The spend decomposition (0:30)**

> Our biggest win wasn't a better model — it was recovering a number the data never states: how much a customer actually spent. It falls out of one accounting identity between balances and payments. That single feature dropped our hidden-test log loss by 0.00173, the largest gain of the project.

**4. Experimentation timeline (0:55)**

> We logged forty-one experiments; only four survived a strict noise threshold. Our score fell from 0.41260 to 0.40982 across eight submissions. One caught us out: isotonic calibration looked like our best result until we found a leakage bug — fixed, it became our worst.

**5. How the model actually predicts (1:15)**

> Take one real customer: late the last two months, maxed out, paid under five percent of a bill. LightGBM — hundreds of decision trees voting in sequence — scores them sixty-three percent. Our GRU reads the six months in order, updating a memory as it goes: thirty-four percent. TabPFN, pretrained and never fit to our data, just shown nineteen thousand labeled customers as context: thirty-one percent. Blended: forty-six percent. That customer defaulted.

**6. Model selection & validation (1:55)**

> That's the whole ensemble: LightGBM at forty-five percent weight, the GRU at thirty, TabPFN at twenty-five — weights chosen by grid search on out-of-fold log loss, not guessed. We validated with repeated stratified five-fold cross-validation across six partitions, moving from 0.42136 out-of-fold to 0.40982 on the hidden test set.

**7. What drives risk (2:20)**

> Interpreting with SHAP: the top predictors are all about recency of lateness — months since last late, the most recent repayment status, whether they were late recently — more than raw balances. Our highest-risk customer scored 85%; our lowest scored just 2%, driven by six clean months and healthy available credit.

**8. Where the model fails (2:45)**

> Two honest failure modes. The "other" education group — created by our own cleaning step — is over-predicted 61%, with an AUC of just 0.645 versus about 0.79 elsewhere. And our twenty worst-predicted customers were all real defaulters scored at 2–4% risk, every one with a spotless payment history. That signal isn't in these columns.

**9. From model to decision (3:15)**

> A false negative extends credit that becomes bad debt — a direct loss. A false positive declines a good customer — lost revenue, plus a fairness cost given that "other" miscalibration. So: auto-approve low risk, auto-decline only the extreme tail, and route everyone else — including that fragile subgroup — to human review.

**10. Limitations & close (3:40)**

> LightGBM alone gets us most of the way; the full ensemble buys a small gain for real reproducibility cost. The levers we can name are exhausted — a better score needs a new insight about the data, not more tuning.

---

### Delivery notes

- The numbers judges are most likely to probe live: **0.40982** (final), **0.00173** (spend-decomposition gain), **62.7% / 34.3% / 31.2% → 46.3%** (the worked customer example), **0.645** (the "other" AUC). Know these cold without looking at a slide.
- If running long, cut from slide 4 (timeline) first, then slide 6 (validation) — its headline numbers already appear on slide 5. Do not cut slide 5 (this is now the deck's answer to "how does the model actually work"), slide 7, or slide 8 — those are the stream's explicit judging questions (mechanism, where it fails, real-world use).
