# Prudential_Life_Insurance_Assessment
## Overview
*A Kaggle Competition*  
In a one-click shopping world with on-demand everything, the life insurance application process is antiquated. Customers provide extensive information to identify risk classification and eligibility, including scheduling medical exams, a process that takes an average of 30 days.

The result? People are turned off. That’s why only 40% of U.S. households own individual life insurance. Prudential wants to make it quicker and less labor intensive for new and existing customers to get a quote while maintaining privacy boundaries.

By developing a predictive model that accurately classifies risk using a more automated approach, you can greatly impact public perception of the industry.

The results will help Prudential better understand the predictive power of the data points in the existing assessment, enabling us to significantly streamline the process.

## TODO
1. [16/12/2015 - **Done**] Imputation
2. [16/12/2015 - **Done**] Sort out the classes of columns
3. [18/12/2015 - **Not Done Yet**] Log transform the features if needed
4. [06/01/2016 - **Done**] try to only use optim within k (fold), not s (loop)
5. [06/01/2016 - **Done**] try to apply optim on valid1 and use it to predict valid2 (remember to change the training set proportion to .66 from .9)
6. [07/01/2016 - **Not Done Yet**] add some noise into it
7. [07/01/2016 - **Done**] tune rf
8. [09/01/2016 - **Done**] knn meta features (need dummy var / binary encode). Failed in xgb, rf, and lr.
9. [09/01/2016 - **Not Done Yet**] try train offset
10. [09/01/2016 - **Done**] pca meta features (need dummy var)
11. [12/01/2016 - **Not Done Yet**] try simple continuous variable transformation (cube, square, log)
12. [30/01/2016 - **Not Done yet**] do tnse and near zero var and personalised fval
13. [01/02/2016 - **Not Done yet**] do distance and try offset
14. [05/02/2016 - **Not Done yet**] feval with integer, rather numeric
15. [05/02/2016 - **Not Done yet**] feature engineering

## LOG
1. I did 3 kinds of imputations (...Impute_Median/Impute_Mean; ...Impute_2016/Impute_1; simply remove the feature)
    + **...Impute_Mean**: Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5.
    + **...Impute_1**: Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5.
    + **...Impute_Median**: Medical_History_1, Medical_History_15.
    + **...Impute_2016**: Medical_History_1, Medical_History_15.
    + **simply remove**: Medical_History_10, Medical_History_24, Medical_History_32.
2. I did 2 kinds of categorical encoding
    + **binary encoding**: for levels > 3 - Product_Info_2, Product_Info_3, Employment_Info_2, InsuredInfo_3, Medical_History_2 
    + **one hot encoding**: for levels == 3 (now in factor class) - Product_Info_7, InsuredInfo_1, Insurance_History_2, Insurance_History_3
    , Insurance_History_4 Insurance_History_7 Insurance_History_8 Insurance_History_9
    , Family_Hist_1, Medical_History_3, Medical_History_5, Medical_History_6, 
    , Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11 
    , Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16 
    , Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20 
    , Medical_History_21, Medical_History_23, Medical_History_25, Medical_History_26 
    , Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30 
    , Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35 
    , Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39 
    , Medical_History_40, Medical_History_41
    + **one hot encoding**: for level == 2 (now in integer class) - Product_Info_1, Product_Info_5, Product_Info_6
    , Employment_Info_3, Employment_Info_5
    , InsuredInfo_2, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7 
    , Insurance_History_1, Medical_History_4, Medical_History_22 
3. There is a technical debt which I threw away all the nzv at once (version 1). A version 2 solution needs to be performed later on where these nzvs will be selected.
4. There is a technical debt for optimising QW kappa based on valid data/cv.
5. xgb performs better when without age, ht, wt, and BMI groups.
6. xgb on raw features, while rf on new features (with the above 4 group features).

## Initial Thoughts
1. Add a feature about the No. of NAs in each section, e.g. NUM_OF_NAS_EMPLOYMENT, NUM_OF_NAS_FAMILY.

## Winners Idea
### 1st
Feature engineering:

create dummy vars for Product_Info_2 (keep everything else as numeric)
calculate sum of all Medical_Keyword columns
for each binary keyword-value pair, calculate the mean of the target variable, then for each observation take the mean and the minimum of the keyword-value-meantargets
Modeling:

for i in 1 to 7: build an XGBClassifier to predict the probability that the observation has a Response value higher than i (for each of the seven iterations, the keyword-value-meantarget variables were calculated for that target variable)
for each observation, take the sum of these seven predicted probabilities as the overall prediction
this yields quite a bit better correlation with the target variable (and thus good raw material for calibration) than using an XGB regressor
Calibration:

the aim is to find the boundaries that maximize the kappa score
boundaries are initialized according to the original Response distribution of the training dataset
then in a step, for all boundaries, possible boundary values are examined in a small range around the current boundary value and the boundary is set to the value which gives the most improvement in kappa (independently of the other boundaries - this was surprising that it worked so well)
steps are repeated until none of the boundaries are changed during a step
it is a quite naive algorithm, but it turned out to be fairly robust and efficient
this was done on predictions generated by repeated crossvalidation using the XGBClassifier combo
Variable split calibration:

the difference here is that the crossvalidated preds are split into two subsets, based on some binary variable value (eg. a Medical_Keyword variable) of the observations
calibration then takes place for the two subsets separately (but with a kappa objective calculated over the entire set), in the manner described above
I didn't find an exact rule for picking a good splitting variable (strong correlation with Response seems to be necessary, but does not guarantee a good split), so I tried several (some of which were better than non-splitting calibration, others were worse)
for example, some good ones were: Medical_History_23, Medical_History_4, InsuredInfo6
also tried splitting into more than 2 subsets, without much success
Ensembling:

disregarding the combination of the 7 XGBClassifiers, the only ensembling I did was creating some combined solutions by taking the median predictions of a small number of other solutions
Evaluating calibrations:

K-fold crossvalidation, but with an important twist: each test fold was "cross-validated" again to imitate public/private test set split (the inner crossvalidation had a k of 3 to approximate the 30-70 leaderboard split)
this yielded a very interesting insight: given two calibrations with roughly equal average performance (over all folds), if calibration A does better on the public test set, calibration B is very likely to outperform A on the private set (this appears to be a quirk of the kappa metric)
accordingly, I picked the solutions which ranked #2 and #5 on the public leaderboard, since these both had very strong average performance in crossvalidation but slightly underperformed on the public leaderboard
Final results:

as it turned out, despite having the right idea about public/private error, I underestimated some solutions which had relatively weak average performance in crossvalidation but ended up doing extremely well on private
I did not select my best private submission for the final two (highest private score was 0.68002)
out of my 11 'high-tech' (that is, using all the modeling and calibration techniques listed above) submissions, 5 were good enough for 1st place on the private board, 4 would place 2nd, one would reach 6th, and the worst would yield 7th place (at least I can say that I had no intention of picking any of the latter two)
if my calculations are right, randomly selecting two out of the 11 would have resulted in 1st place with a probability of ~72.7 %  

### 2nd
1. Initial Feature Engineering (taken from public scripts):

all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]
all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]
all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']
med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)
all_data.apply(lambda x: sum(x.isnull()),1)
all_data['countna'] = all_data.apply(lambda x: sum(x.isnull()),1)
2. Next step was to calculate prediction for y=1,2,3,4,5,6,7,8 and y<3,<4,<5,<6,<7.

To calculate these probabilities I used an ensemble from one binary:logistic Xgboost, one multi:softprob Xgboost, one Random Forest and one LogisticRegression (I tried a lot of ensembles but they didn't help). So overall it added me 13 additional features.

3. Linear Regression!

Here was the place when mystery began! I tried a huge amount of regressors and ensembles of them but they never beat the simplest Linear Regression! That puzzle really got me and made me confused for a few days until I realized that all "work" is done during probability calculations. So I decided to concentrate on them hopefully that my model at least won't overfit due to linear model.

4. The function to search for cutoffs. Until last day I used the cutoff function which suggested someone on the forum. It's the one with these code:

### 3rd
attributes

base attributes, number of keywords, a few other things suggested on the forum
2D tnse embedding
the 2D embedding generated by a 4096-256-16-2-16-256-4096 autoencoder
The first 30-dimensions of a SVD decomposition of the categorical features
kmeans clustering with 64 centers
quadratic interactions selected by lasso mse regression
nodes of a 256-tree 50-node random forest selected by lasso mse regression
level 1 models

tree based models: 8 xgboost models minimizing: mse, possion, multinomial, mae*, tukey*, or QWK* loss.
knn: 8 k-nearest neighbor models with k from 50-1000
neural nets: 6 neural networks minimizing: mse, mae, multinomial, or QWK* loss
linear: 1 lasso mse regression
level 2 models

both use the level 1 models as inputs
multinomial xgb
multinomial neural net
QWK optimization: Uses the average of the level 2 models as the class probabilities.

randomly select a category 1-8 for every test example
iterate through the test examples one by one and change each class to whatever maximizes the expected value of the QWK based on the probabilites of the level 2 models.
repeat step 2. over the entire test set until convergence is reached
submit result












