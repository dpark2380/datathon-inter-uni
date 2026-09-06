# Summary of Final Approach

Our task was to predict the probability that a credit card customer defaults on
their next payment, scored on binary log loss over a 6,000-customer test set. Because
the metric rewards well-calibrated probabilities rather than correct labels, every
decision we made attempted to further optimize log-less instead of other metrics (e.g. AUC).

Our final submission is a weighted average of three models:

- **LightGBM (0.45)** on 81 engineered features — the aggregate view 
- **Bidirectional GRU (0.30)** reading the six monthly statements in order — the trajectory view
- **TabPFN (0.25)**, a transformer pre-trained on synthetic tabular data, which predicts in-context without fitting to our data at all

This scored **0.40982** on the hidden test set. We arrived at it through 41 logged
experiments, of which only four produced a gain that survived cross-validation. The
two things that actually worked were deriving information the dataset never stated
(e.g. monthly spending, which was recovered from an accounting identity) and averaging many models
to reduce variance. Seven other model families and six other feature families were
tested and rejected based on metrics.

**Calibration:** we tested four methods (Platt scaling, isotonic regression, per-segment calibration, and base-rate shrinkage)
and recorded no improvements and thus did not apply it to our final solution. This is because the probabilities we obtained were already well-calibrated.

# Data Cleaning and Preprocessing
We opened with preprocessing and cleaning to reduce the amount of redundant information in the dataset, and this was done primarily by identifying the meaning of each source-coded predictor.
1. **Repayment Status**  
   When looking at the variables regarding repayment status (PAY_0, PAY_2, ... , PAY_6), we noted that there were more unique number entries than expected. In our EDA, we broke it down as such:

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
   | 9 | Payment has been delayed by 9+ months (Not in training set) |

   We noted that the repayment status is important in providing information about whether or not this customer has a good history of repaying their loans on time and could be a very good indicator of default. As the number increased, naturally, the customer's expectation of repaying their loans decreases. However, at the very top end with values (-2, -1, 0), they all carry the same meaning, namely that the customer had a good record for the given month. As such, we group all of these values together to 0.

   We did this during feature engineering, and got different results when compared to doing it during cleaning.
   The raw `PAY_*` columns are left untouched in `train_clean.csv`; the lateness features are computed on
   `PAY.clip(lower=0)`, which folds -2 and -1 into 0, while `ever_paid_full` and `never_used` read the original
   -1 and -2 codes. If we collapsed the codes in the cleaned file itself, those two
   features could not exist.

3. **Marriage**  
We noted that within this dataset, the values for marriage were defined as such. 
    - **0**: Unknown
    - **1**: Married
    - **2**: Single
    - **3**: Other  
    
    However, for the purposes of training our model and making predictions using this variable, the effect of unknown vs other felt noisy as opposed to something actionable. Therefore, we made all values of 0 equal to 3 instead.

4. **Education**  
    The education column has 7 variables (0, 1, 2, 3, 4, 5, 6). However, only the following are officially defined:  
    - **1**: Graduate School
    - **2**: University
    - **3**: High School
    - **4**: Other  

    As such, we grouped everything classified as 0, 5 or 6 into the other category.

With these features cleaned, we exported the cleaned training set as
```datasets/train_clean.csv```. The same two substitutions are implemented in code as
`common.clean()`, implemented by every tested model.

```bash
uv run python -c "import sys, pandas as pd; sys.path.insert(0, 'models'); \
from common import clean; clean(pd.read_csv('datasets/train.csv')).to_csv('datasets/train_clean.csv', index=False)"
```

(`2_Column_Inspection.ipynb` is where we worked out which substitutions were needed. it
prints the unique values and null counts per column. The substitutions themselves live
in `common.clean()` so that train and test sets are treated identically.)

# Feature Engineering
After training a number of models on our cleaned dataset, we ended up hitting a wall in our final log loss results. We therefore turn to feature engineering to identify potentially hidden patterns.
1. **Repayment Status History**  
The first thing we changed was the repayment status history. We thought that the repayment status history was one of the most important variables that could affect our prediction, yet it wasn't detailed enough in the given data. As such, we broke it down from simply the number of months delayed they were to a set of much more detailed figures.  

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
Here we made a number of other features, namely ```util1``` to ```util6```, ```util_mean```, ```util_max```, ```util_trend``` and ```avail_credit```. These represent how much of their available credit each consumer is using up each month and whether that trend is increasing or decreasing. These are useful features as consistently high untilisation suggests that the individual may be lacking funds, and thus borrows more from the bank.  
3. **Payment Coverage**  
These features (```payratio1``` to ```payratio5```) was calculated by dividing the payment amount in a given month by the amount owed in the previous month (the amount they are paying off). We believed this to be a useful indicator as we expect individuals who pay off a higher proportion of their loans to be less likely to default.  
4. **Absolute Levels and Momentum**  
These features are included in ```bill_sum```, ```bill_mean```, ```bill_std```, ```amt_sum```, ```amt_mean```, ```amt_std```, ```coverage_total```, ```bill_growth```, ```amt_over_limit``` and ```log_limit```.

    Every feature family above this one is a ratio or a count, which deliberately
    strips out scale. `payratio3` is the same number whether the customer owes
    \$500 or \$50,000. That is usually what we want, but the downside is that it neglects accounting for
    the individual's debt status.

    | Feature | What it captures |
    |---------|------------------|
    | ```bill_sum```, ```bill_mean```, ```bill_std``` | The overall size and volatility of what they owe |
    | ```amt_sum```, ```amt_mean```, ```amt_std``` | The same for what they actually pay |
    | ```coverage_total``` | Total paid ÷ total billed across all six months (overall repayment rate) |
    | ```bill_growth``` | Balance in the most recent month minus the oldest (net debt change across window) |
    | ```amt_over_limit``` | Total paid relative to their credit limit (payment volume scaled to their line) |
    | ```log_limit``` | Log of the credit limit, which compresses a very heavy-tailed variable |

    We added these alongside our MLP. Since a decision tree is scale-invariant, multiplying a feature by 1,000 or taking its logarithm changes
    nothing about which splits it finds, and thus the above is unnecessary. A neural network is not, so a feature measured in tens of thousands
    will swamp one measured in fractions unless the inputs are standardised and
    the magnitudes are expressed in a form it can use. These features
    provide the network with scale information.
6. **Spending Decomposition**  
These features are listed under ```spend1``` to ```spend5```, ```spend_mean```, ```spend_max```, ```spend_std```, ```spend_trend```, ```spend_total```, ```n_months_no_spend```, ```spend_minus_paid```, ```months_spent_gt_paid```. We calculated the spend amount by subtracting the difference in bills between two months and then adding back the amonut which the customer paid off. We added this as our models treated the bill amount features with quite high importance. However, without directly laying out their spending, it was impossible to identify whether a customer's bill amount was going up because they were spending more or because they were not paying back their debts, a difference which told two very different stories.
7. **Minimum Payment Behaviour**  
For this, we added the features ```min_pay_ratio_mean```, ```min_pay_ratio_min```, ```months_paid_about_min```, ```months_paid_under_min```. We used this to flag whether or not a customer was paying only (or around) the minimum amount every month. This, in general, can be taken as a sign of trouble, especially is the individual is borrowing a large amount, generally indicating a larger need for month which they are unable to afford.

# Validation Strategy
Because the competition is scored on log loss over a hidden test set, we needed a
local estimate we could trust before spending a submission. We used stratified
5-fold cross-validation, repeated over 6 different fold partitions, with every
model calling the same function and the same seeds for direct OOF comparisons.

## Rationale

1. We established a noise floor by repeating experiments on identical splits. Through this we
found that any difference below 0.0005 in OOF log loss is indistinguishable
from fold noise on 24,000 rows. We report every experiment against that threshold and
never adopted a change below it on its point estimate alone.
3. All fitted models during validation used OOF samples. Blend weights, calibrators
etc were measured on rows the fitting was blind to. We adapted this strategy after catching it in a previous iteration (see Limitations below0.
4.  Repeated CV and full-data refit cannot
show up in OOF scoring by construction, since each OOF row is predicted only by its
own fold's models.

The relationship between the two stayed stable throughout (OOF ≈ 0.4214 against a hidden
test score of ≈ 0.4098), so we used OOF to decide whether to adopt something and the
leaderboard only to confirm it.

# Models Tested and Final Model Selection
10 were tested and 3 were used in the final weighted blend:

| Kept | OOF log loss | Weight | Rationale |
|------|--------------|--------|--------------------------|
| LightGBM | 0.42211 | 0.45 | Strongest single model; reads 81 features as flat description of a customer |
| Bidirectional GRU | 0.42392 | 0.30 | Reads the six months in order, separating customers the aggregates cannot |
| TabPFN | 0.42345 | 0.25 | A pretrained model which matched our tuned GRU with no tuning at all |

We use an example to support why a blend performed the best.
Take two customers who were each late twice. One was late five and six
months ago and has paid cleanly since, and the other paid cleanly until missing the last two
months. Their `n_late`, `pay_max` and `pay_mean` are identical, and summary features
genuinely cannot tell them apart, even though they're trending completely differently.
However, reading the months in order can, which GRU provides here. Through this we make more confident predictions on top of accuracy and thus improve log-loss.

Everything else was rejected on measurement: CatBoost, ExtraTrees, logistic regression,
1D-CNN, autoencoder, multi-task GRU, survival/hazard model and our original MLP
all received weight 0.00 in the final weight search. The pattern across all of them is
that a model which is a different algorithm over the same features adds nothing.
Thus, we used models which processed training data differently in the final blend.

We also ran 135 LightGBM configs across three hyperparameter sweeps resulting in negligible improvement.

# Ensembling and Post-Processing
Our three models are combined by a weighted average of their probabilities, since meta-models and per-customer blends performed much worse.

The weights were chosen by grid search over the weight simplex in 0.05 steps, scored on
OOF log-loss. The surface turned out to be very flat, where every weighting from 0.50
to 0.70 on LightGBM sat within 0.0001. This meant that the exact choice barely matters and the weights weren't finely tuned to
noise. We confirmed this by shrinking them halfway toward equal (to account for overfitting), and the score got slightly worse.

**We apply no calibration.** We tested Platt scaling, isotonic regression, per-segment calibration and shrinkage toward the base rate. As stated above, the output from our blend model was already well-calibrated. All four made log loss worse under proper nested cross-validation, and the shrinkage search independently landed on "do not adjust at all". A reliability diagram confirms the same thing from the other direction, where our predicted probabilities track observed default rates across every decile. 

The only post-processing we do is clipping the final probabilities to `[1e-4, 1-1e-4]`. This is insurance and we did not tune it. Log-loss heavily penalises a single confidently-wrong prediction, and the clip caps that error penalty.

# Key Results and Observation

Our final submission scored **0.40982** on the hidden test set, against 0.42136 OOF.

Of 41 logged experiments, only four produced a gain that survived. The largest by far came from an accounting identity. The data gives us statement balances and payments but never the amount a customer actually charged, and since `BILL_t = BILL_(t+1) - PAY_AMT_t + spend_t`, we could recover it. That single feature family was worth −0.00173 on the hidden test set. Everything else that helped was variance reduction, namely repeated cross-validation and full-data refits.

We found that new information wins, but new algorithms don't. Seven model families were rejected outright while one accounting identity produced our biggest gain. The corollary surprised us, because decorrelation on its own turned out to be insufficient. Logistic regression was the most decorrelated model we produced (0.948 against LightGBM) and still earned weight 0.00, while the GRU earned 0.30 at a higher correlation of 0.9825. What matters is whether the disagreement carries information, and the mere existence of disagreement tells us nothing.

We identified a mistake prior to our final model, which was that isotonic calibration initially seemed beneficial. It was being fitted on the same OOF predictions it was then scored against, and a flexible monotonic function absorbed the noise in the rows judging it. We spotted this, and re-ran under proper nested CV, and results were immediately different. Following this every fitted combiner after that point was scored strictly OOF.

**Limitations**

1. Our twenty worst-predicted customers were outlying defaulters that the model rated at 2 to 4% risk, and every one of them has a good payment history, never late, moderate utilisation, healthy limits.

2. Through an audit we found `EDUCATION="other"` (387 customers, 28 defaults) is over-predicted by 61% in relative terms and ranked poorly, at AUC 0.645 against roughly 0.79 elsewhere. This is due to our data cleaning, where we collapsed these categories. It is also the only segment where our model does worse than predicting that group's base rate.

3. LightGBM alone reaches 0.42211 OOF and all three together reach 0.42136, thus computing our blend through heavy computation wasn't an efficient/high return process.
