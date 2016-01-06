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
6. [07/01/2016 - **Not Done Yet**] add some noise into it.

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

## Initial Thoughts
1. Add a feature about the No. of NAs in each section, e.g. NUM_OF_NAS_EMPLOYMENT, NUM_OF_NAS_FAMILY.