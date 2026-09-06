# Summary of Final Approach

Our task was to predict the probability that a credit card customer defaults on
their next payment, scored on binary log loss over 6,000 hidden customers. Because
the metric rewards well-calibrated probabilities rather than correct labels, every
decision we made was judged on log loss directly, never on accuracy or AUC.

Our final submission is a weighted average of three models that read the same
customer in three different ways:

- **LightGBM (0.45)** on 81 engineered features — the aggregate view of a customer
- **A bidirectional GRU (0.30)** reading the six monthly statements in order — the trajectory view
- **TabPFN (0.25)**, a transformer pretrained on synthetic tabular data, which predicts in-context without fitting to our data at all

This scored **0.40982** on the hidden test set. We arrived at it through 41 logged
experiments, of which only four produced a gain that survived cross-validation. The
two things that actually worked were deriving information the dataset never stated
(monthly spending, recovered from an accounting identity) and averaging many models
to reduce variance. Seven other model families and six other feature families were
tested and rejected on measurement, not intuition.

We deliberately applied no calibration. We tested four methods — Platt scaling,
isotonic regression, per-segment calibration and base-rate shrinkage. Three made the
score worse and the fourth selected "no adjustment" on its own, because the
probabilities were already well calibrated.

# Data Cleaning and Preprocessing
Before we started the task, we opened with preprocessing and cleaning to try to reduce the amount of useless information floatig around inside the dataset. By performing a deep dive into each of the variables and the meaning and information they represent, we identified a few pieces of data cleaning which we performed.
1. **Repayment Status**  
   When looking at the variables regarding repayment status (PAY_0, PAY_2, ... , PAY_6), we noted that there were more unique number entries that we would expect. In our EDA, we broke it down as such.

   | Values | Description |
   |--------|-------------|
   | -2 | No Balance / No Credit Used |
   | -1 | Paid in Full -> Customer completely paid off the charge on the card |
   | 0 | Paid Minimum Amount -> Customer used the card and paid off the min amount |
   | 1 | Payment has been delayed by 1 month | 
   | 2 | Payment has been delayed by 2 months | 
   | 3 | Payment has been delayed by 3 months | 
   | 4 | Payment has been delayed by 4 months |
   | 5 | Payment has been delayed by 5 months | 
   | 6 | Payment has been delayed by 6 months |
   | 7 | Payment has been delayed by 7 months |
   | 8 | Payment has been delayed by 8 months |
   | 9 | Payment has been delayed by 9+ months (This value doesn't appear in the training dataset) |

   We noted that the repayment status is important in providing information about whether or not this customer has a good history of repaying their loans on time and could be a very good indicator of default. As the number increased, naturally, the customer's expectation of repaying their loans decreases but at the very top end with values (-2, -1, 0), they all carry the same meaning, namely that the customer has a good record this month. As such, we group all of these values together to be equal to 0.

   We do this **inside feature engineering rather than in the cleaning step**, and the
   distinction turned out to matter. The raw `PAY_*` columns are left untouched in
   `train_clean.csv`; the lateness features are computed on `PAY.clip(lower=0)`, which
   folds -2 and -1 into 0, while `ever_paid_full` and `never_used` read the original
   -1 and -2 codes. Had we collapsed the codes in the cleaned file itself, those two
   features could not exist. Doing it per-family keeps both readings available.

2. **Marriage**  
We noted that within this dataset, the values for marriage were defined as such. 
    - **0**: Unknown
    - **1**: Married
    - **2**: Single
    - **3**: Other  
    
    However, for the purposes of training our model and making predictions using this variable, a marriage status of unknown is no different from a marriage status of other. Therefore, we made all values of 0 equal to 3 instead.

3. **Education**  
    The education column has 7 variables (0, 1, 2, 3, 4, 5, 6). However, offically only the following are defined:  
    - **1**: Graduate School
    - **2**: University
    - **3**: High School
    - **4**: Other  

    As such, we grouped everything classified as 0, 5 and 6 into the other category.  

With these features cleaned, we exported the cleaned training set as
```datasets/train_clean.csv```. The same two substitutions are implemented in code as
`common.clean()`, which every model calls, so the cleaned file is exactly reproducible:

```bash
uv run python -c "import sys, pandas as pd; sys.path.insert(0, 'models'); \
from common import clean; clean(pd.read_csv('datasets/train.csv')).to_csv('datasets/train_clean.csv', index=False)"
```

(`2_Column_Inspection.ipynb` is where we worked out *which* substitutions were needed — it
prints the unique values and null counts per column. The substitutions themselves live
in `common.clean()` so that train and test are treated identically.)

# Feature Engineering
After training a number of models on our cleaned dataset, we ended up hitting a wall in our final log loss results. As such, we turned to feature engineering to create more useful variables and predictors that we could train our model on.  
1. **Repayment Status History**  
The first thing we changed was the repayment status history. We thought that the repayment status history was one of the most important variables that could affect our prediction yet, it wasn't detailed enough in the data given. As such, we broke it down from simply the number of months delayed they were to a set of much more detailed figures.  

   | Feature | Description |
   |--------|-------------|
   | ```pay_max```, ```pay_sum```, ```pay_mean```, ```pay_std``` | Basic summary of how late a customer has been in the past|
   | ```n_late```, ```n_late2plus```| Number of months late and number of seriously late months (we believed anything over 2 months late is serious)|
   | ```trend_pay``` | An indicator describing whether they have been getting better or worse at repaying over the past 6 months|
   | ```recent_late``` | A score of how late they are weighted by time. (More recent lateness is weighted heavier) | 
   | ```ever_paid_full```, ```never_used``` | Brought back the two values -2 (never used) and -1 (paid off loan in full) | 
    | ```max_late_streak``` | Longest consecutive run of late months | 
   | ```month_since_late``` | Amount of months since they were last late on a payment |  

2. **Credit Utilisation**  
Here we made a number of other features namely ```util1``` to ```util6```, ```util_mean```, ```util_max```, ```util_trend``` and ```avail_credit```. These represent how much of their available credit each consumer is using up each month and whether that trend is increasing or decreasing. These are useful features as consistently high untilisation suggests that the individual may be lacking funds, and thus borrows more from the bank.  
3. **Payment Coverage**  
These features (```payratio1``` to ```payratio5```) was calculated by dividing the payment amount in a given month by the amount owed in the previous month (the amount they are paying off). We believed this to be a useful indicator as we expect individuals who pay off a higher proportion of their loans to be less likely to default.  
4. **Absolute Levels and Momentum**  
These features are included in ```bill_sum```, ```bill_mean```, ```bill_std```, ```amt_sum```, ```amt_mean```, ```amt_std```, ```coverage_total```, ```bill_growth```, ```amt_over_limit``` and ```log_limit```.

    Every feature family above this one is a ratio or a count, which deliberately
    strips out scale — `payratio3` is the same number whether the customer owes
    \$500 or \$50,000. That is usually what we want, but it throws away something
    real: the absolute size of someone's debt, and whether it is growing.

    | Feature | What it captures |
    |---------|------------------|
    | ```bill_sum```, ```bill_mean```, ```bill_std``` | The overall size and volatility of what they owe |
    | ```amt_sum```, ```amt_mean```, ```amt_std``` | The same for what they actually pay |
    | ```coverage_total``` | Total paid ÷ total billed across all six months — their overall repayment rate |
    | ```bill_growth``` | Balance in the most recent month minus the oldest — net debt change across the window |
    | ```amt_over_limit``` | Total paid relative to their credit limit — payment volume scaled to their line |
    | ```log_limit``` | Log of the credit limit, which compresses a very heavy-tailed variable |

    We added these alongside our MLP, and the reason is a real difference between
    the two model types. A decision tree is **scale-invariant**: it splits on
    thresholds, so multiplying a feature by 1,000 or taking its logarithm changes
    nothing about which splits it finds. A neural network is not — it multiplies
    inputs by weights and sums them, so a feature measured in tens of thousands
    will swamp one measured in fractions unless the inputs are standardised and
    the magnitudes are expressed in a form the network can use. These features
    give the network the scale information the ratio features had removed.
5. **Spending Decomposition**  
These features are listed under ```spend1``` to ```spend5```, ```spend_mean```, ```spend_max```, ```spend_std```, ```spend_trend```, ```spend_total```, ```n_months_no_spend```, ```spend_minus_paid```, ```months_spent_gt_paid```. We calculated the spend amount by subtracting the difference in bills between two months and then adding back the amonut which the customer paid off. We added this as our models treated the bill amount features with quite high importance. However, without directly laying out their spending, it was impossible to identify whether a customer's bill amount was going up because they were spending more or because they were not paying back their debts, a difference which told two very different stories.
6. **Minimum Payment Behaviour**  
For this, we added the features ```min_pay_ratio_mean```, ```min_pay_ratio_min```, ```months_paid_about_min```, ```months_paid_under_min```. We used this to flag whether or not a customer was paying only (or around) the minimum amount every month. This, in general, can be taken as a sign of trouble, especially is the individual is borrowing a large amount, generally indicating a larger need for month which they are unable to afford. 
# Validation Strategy
Because the competition is scored on log loss over a hidden test set, we needed a
local estimate we could trust before spending a submission. We used **stratified
5-fold cross-validation, repeated over 6 different fold partitions**, with every
model calling the same function and the same seeds so their out-of-fold predictions
line up row-for-row and can be compared directly.

Three things about this mattered more than the split itself.

1. **We established a noise floor.** By repeating experiments on identical splits we
found that any difference below **0.0005** in out-of-fold log loss is indistinguishable
from fold noise on 24,000 rows. We report every experiment against that threshold and
never adopted a change below it on its point estimate alone.
2. **Anything we fitted was scored strictly out-of-fold.** Blend weights, calibrators
and any target-derived feature are measured on rows the fitting never saw. This was not
our original protocol — we got caught by it once, described under Limitations below.
3. **We knew where our estimate was blind.** Repeated CV and the full-data refit cannot
show up in out-of-fold scoring by construction: each OOF row is predicted only by its
own fold's models, and OOF never involves a model refitted on all the data. We confirmed
this on the leaderboard — the full-data refit was worth 0.00023 there and exactly zero
on OOF.

The relationship between the two stayed stable throughout (OOF ≈ 0.4214 against a hidden
test score of ≈ 0.4098), so we used OOF to decide *whether* to adopt something and the
leaderboard only to confirm it.

# Models Tested and Final Model Selection
We tried **ten model families. Three earned a place in the final blend.**

| Kept | OOF log loss | Weight | Why it earned its place |
|------|--------------|--------|--------------------------|
| LightGBM | 0.42211 | 0.45 | Our strongest single model; reads the 81 features as a flat description of a customer |
| Bidirectional GRU | 0.42392 | 0.30 | Reads the six months **in order**, separating customers the aggregates cannot |
| TabPFN | 0.42345 | 0.25 | A pretrained prior — different in kind, and it matched our tuned GRU with no tuning at all |

The GRU is worth explaining, because it is the clearest example of why we needed more
than one model. Take two customers who were each late twice. One was late five and six
months ago and has paid cleanly since; the other paid cleanly until missing the last two
months. Their `n_late`, `pay_max` and `pay_mean` are **identical** — our summary features
genuinely cannot tell them apart, even though one is recovering and one is falling apart.
Reading the months in order can.

Everything else was rejected on measurement: CatBoost, ExtraTrees, logistic regression,
a 1D-CNN, an autoencoder, a multi-task GRU, a survival/hazard model and our original MLP
all received **weight 0.00** in the final weight search. The pattern across all of them is
that a model which is merely a *different algorithm* over the same features adds nothing —
the CNN correlated 0.9924 with the GRU, so it split the sequence weight rather than adding
to it. Only models that read the data a genuinely different way earned anything.

We also ran **135 LightGBM configurations** across three hyperparameter sweeps. Our
incumbent settings were never beaten; the best challenger was 0.00004 better, twelve times
below our noise floor.

# Ensembling and Post-Processing
Our three models are combined by a **plain weighted average of their probabilities** —
no meta-model, no per-customer switching. We tested both of those and they were among the
worst results we recorded, with a feature-aware meta-model coming in 0.00208 *worse* than
simple averaging.

The weights were chosen by grid search over the weight simplex in 0.05 steps, scored on
out-of-fold log loss. The surface turned out to be very flat — every weighting from 0.50
to 0.70 on LightGBM sits within 0.0001 — which is reassuring rather than disappointing,
because it means the exact choice barely matters and the weights are not finely tuned to
noise. We confirmed that by shrinking them halfway toward equal, the standard remedy for
overfitted weights, and the score got slightly *worse*.

**We apply no calibration**, and that is a finding rather than an omission. We tested
four methods, each under proper nested cross-validation:

| Method | What it does | Result |
|--------|--------------|--------|
| Platt scaling | Fits a two-parameter sigmoid to the predicted probabilities | 0.42139, worse by 0.00004 — essentially a no-op |
| Isotonic regression | Fits a free monotonic step function | 0.42508, worse by 0.00372 — the worst result we recorded |
| Per-segment calibration | Separate curves by credit-limit quartile and lateness count | worse by 0.00030 |
| Base-rate shrinkage | A one-parameter pull toward the 22.12% base rate | the search selected α = 1.00, i.e. "do not adjust at all" |

Three made log loss worse; the fourth declined to change anything. A reliability diagram
confirms our predicted probabilities track observed default rates across every decile.

The only post-processing is clipping the final probabilities to `[1e-4, 1-1e-4]`, which
is insurance against a single confidently-wrong prediction contributing an enormous
penalty under log loss.

# Key Results, Observations and Limitations
Our final submission scored **0.40982** on the hidden test set, against 0.42136
out-of-fold.

**What actually worked.** Of 41 logged experiments, four produced a gain that survived.
The largest by far came from an accounting identity: the data gives us statement balances
and payments but never the amount a customer actually *charged*, and since
`BILL_t = BILL_(t+1) - PAY_AMT_t + spend_t`, we could recover it. That single feature
family was worth −0.00173 on the hidden test set. Everything else that helped was
variance reduction — repeated cross-validation and full-data refits.

**The clearest lesson.** New *information* wins; new *algorithms* do not. Seven model
families were rejected outright, while one accounting identity produced our biggest gain.
The corollary surprised us: decorrelation alone is not enough. Logistic regression was the
most decorrelated model we produced (0.948 against LightGBM) and still earned weight 0.00,
while the GRU earned 0.30 at a *higher* correlation of 0.9825. What matters is whether the
disagreement is informative, not whether it exists.

**A mistake worth recording.** Isotonic calibration initially looked like a large win —
0.41946 against a 0.42136 baseline. It was being fitted on the same out-of-fold
predictions it was then scored against, and a free monotonic function will happily
absorb the noise in the rows judging it. Re-run with proper nested cross-validation it
scored 0.42508, the single worst result in our table. Every fitted combiner after that
point was scored strictly out-of-fold.

**Limitations.**
- **Part of this problem is unreachable from this data.** Our twenty worst-predicted
  customers are all defaulters the model rated at 2–4% risk, and every one of them has a
  spotless payment history — never late, moderate utilisation, healthy limits. Whatever
  caused those defaults is not observable in the columns we were given.
- **One subgroup is served noticeably worse.** Our fairness audit found `EDUCATION="other"`
  (387 customers, 28 defaults) is over-predicted by 61% relative and ranked poorly
  (AUC 0.645 against ~0.79 elsewhere). That category is where our own cleaning collapses
  undocumented codes 0, 5 and 6, so it is a heterogeneous catch-all. It is the only group
  where our model does worse than simply predicting that group's base rate.
- **Most of the value sits in one model.** LightGBM alone reaches 0.42211 out-of-fold; all
  three together reach 0.42136. The other two components buy roughly 0.0007 for 180 extra
  model fits and two additional dependencies.
- **The blend weights are fitted to validation data**, which is the one step in our
  pipeline where genuine selection — and so overfitting risk — occurs.
