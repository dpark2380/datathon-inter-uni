# Copy of Kaggle info


# Overview
Financial institutions need to assess whether customers are likely to meet their repayment obligations. Reliable risk estimates can support account review, credit-limit decisions, and early intervention while avoiding unnecessary restrictions on customers who are likely to repay.

In this competition, you will use anonymised customer credit information, including credit limits, demographic attributes, repayment history, bill amounts, and previous payments, to estimate whether a customer will default on their next payment.

The training set contains 24,000 labelled customers, while 6,000 customers are included in the hidden test set. The split is stratified so that the proportion of default cases remains comparable across both sets.

The target is default:

1 indicates default on the next payment. 0 indicates no default.

Your model must produce a probability of default rather than only a binary decision.

Participants should consider the consequences of different errors. A false negative may expose a lender to unexpected financial loss, while a false positive may lead to unnecessary restrictions for a reliable customer. Participants are also encouraged to consider model calibration, interpretability, and responsible use of demographic information.

# Evaluation
Submissions are evaluated using binary log loss:

$$
\begin{aligned}
f(x) &= 2x^2 + 3x - 5 \\
\frac{df}{dx} &= 4x + 3
\end{aligned}
$$

where:

(`N`) is the number of customers in the hidden test set.
(`y_i`) is the true outcome for customer (i), either 0 or 1.
(`p_i`) is the submitted probability that customer (i) will default.
Lower scores are better.

Log loss rewards accurate and well-calibrated probability estimates. It also strongly penalises confident predictions that turn out to be incorrect.


# Dataset Description
## Data dictionary

`client_id`: non-informative competition identifier.
`LIMIT_BAL`: granted credit in NT dollars.
`SEX`: source-coded sex category.
`EDUCATION`: source-coded education category.
`MARRIAGE`: source-coded marital-status category.
`AGE`: age in years.
`PAY_0, PAY_2 … PAY_6`: recent repayment-status history.
`BILL_AMT1 … BILL_AMT6`: monthly bill statement amounts in NT dollars.
`PAY_AMT1 … PAY_AMT6`: monthly previous-payment amounts in NT dollars.
`default: training target`; 1 = default, 0 = no default.