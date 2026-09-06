# Summary of Final Approach




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

   We noted that the repayment status is important in providing information about whether or not this customer has a good history of repaying their loans on time and could be a very good indicator of default. As the number increased, naturally, the customer's expectation of repaying their loans decreases but at the very top end with values (-2, -1, 0), they all carry the same meaning, namely that the customer has a good record this month. As such, we grouped all of these values together to be equal to 0.

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

With these features cleaned, we then exported the cleaned training model as a csv named ```train_clean.csv```.

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
These features are included in ```bill_sum```, ```bill_mean```, ```bill_std```, ```amt_sum```, ```amt_mean```, ```amt_std```, ```coverage_total```, ```bill_growth```, ```amt_over_limit``` and ```log_limit```. These help quantify the scale of bills and payments. We only added these features later on when we were trying our MLP model as it cared about scale.
5. **Spending Decomposition**  
These features are listed under ```spend1``` to ```spend5```, ```spend_mean```, ```spend_max```, ```spend_std```, ```spend_trend```, ```spend_total```, ```n_months_no_spend```, ```spend_minus_paid```, ```months_spent_gt_paid```. We calculated the spend amount by subtracting the difference in bills between two months and then adding back the amonut which the customer paid off. We added this as our models treated the bill amount features with quite high importance. However, without directly laying out their spending, it was impossible to identify whether a customer's bill amount was going up because they were spending more or because they were not paying back their debts, a difference which told two very different stories.
6. **Minimum Payment Behaviour**  
For this, we added the features ```min_pay_ratio_mean```, ```min_pay_ratio_min```, ```months_paid_about_min```, ```months_paid_under_min```. We used this to flag whether or not a customer was paying only (or around) the minimum amount every month. This, in general, can be taken as a sign of trouble, especially is the individual is borrowing a large amount, generally indicating a larger need for month which they are unable to afford. 