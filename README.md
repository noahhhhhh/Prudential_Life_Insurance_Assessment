# Prudential_Life_Insurance_Assessment
## Overview
*A Kaggle Competition*  
In a one-click shopping world with on-demand everything, the life insurance application process is antiquated. Customers provide extensive information to identify risk classification and eligibility, including scheduling medical exams, a process that takes an average of 30 days.

The result? People are turned off. That’s why only 40% of U.S. households own individual life insurance. Prudential wants to make it quicker and less labor intensive for new and existing customers to get a quote while maintaining privacy boundaries.

By developing a predictive model that accurately classifies risk using a more automated approach, you can greatly impact public perception of the industry.

The results will help Prudential better understand the predictive power of the data points in the existing assessment, enabling us to significantly streamline the process.

## TODO
1. [16/12/2015 - **Done**] Imputation
2. [16/12/2015 - **Not Done Yet**] Sort out the classes of columns
3. [18/12/2015 - **Not Done Yet**] Log transform the features if needed

## LOG
1. I did 3 kinds of imputations (...Impute_Median/Impute_Mean; ...Impute_2016/Impute_1; simply remove the feature)
    + **...Impute_Mean**: Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5.
    + **...Impute_1**: Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5.
    + **...Impute_Median**: Medical_History_1, Medical_History_15.
    + **...Impute_2016**: Medical_History_1, Medical_History_15.
    + **simply remove**: Medical_History_10, Medical_History_24, Medical_History_32.

## Initial Thoughts
1. Add a feature about the No. of NAs in each section, e.g. NUM_OF_NAS_EMPLOYMENT, NUM_OF_NAS_FAMILY.