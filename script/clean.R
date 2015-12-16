rm(list = ls()); gc();
setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Prudential_Life_Insurance_Assessment/")

require(data.table)
source("script/utilities/preprocess.R") # utilities functions for preprocessing data
############################################################################################
## 1.0 read ################################################################################
############################################################################################
dt.raw.train <- fread("data/data_raw/train.csv")
dim(dt.raw.train)
# [1] 59381   128
dt.raw.test <- fread("data/data_raw/test.csv")
dim(dt.raw.test)
# [1] 19765   127
dt.raw.sampleSubmit <- fread("data/data_raw/sample_submission.csv")
dim(dt.raw.sampleSubmit)
# [1] 19765     2

####################################
## 1.1 combine the test and train ##
####################################
# create a dummy response
dt.raw.test[, Response := rep(0, dim(dt.raw.test)[1])]
# create a dummy idnt of test set
dt.raw.test[, isTest := rep(1, dim(dt.raw.test)[1])]
# create a dummy idnt of train set
dt.raw.train[, isTest := rep(0, dim(dt.raw.train)[1])]

# combine the two dts
identical(names(dt.raw.train), names(dt.raw.test))
# [1] TRUE
dt.raw.combine <- rbind(dt.raw.train, dt.raw.test)
dim(dt.raw.combine); dim(dt.raw.train); dim(dt.raw.test)
# [1] 79146   129
# [1] 59381   129
# [1] 19765   129
############################################################################################
## 2.0 inspect #############################################################################
############################################################################################
#############
## 2.1 NAs ##
#############
ColNAs(dt = dt.raw.combine, method = "sum", output = "nonZero")
# Employment_Info_1   Employment_Info_4   Employment_Info_6 Insurance_History_5       Family_Hist_2 
# 22                8916               14641               33501               38536 
# Family_Hist_3       Family_Hist_4       Family_Hist_5   Medical_History_1  Medical_History_10 
# 45305               25861               55435               11861               78388 
# Medical_History_15  Medical_History_24  Medical_History_32 
# 59460               74165               77688
ColNAs(dt = dt.raw.combine, method = "mean", output = "nonZero")
# Employment_Info_4   Employment_Info_6 Insurance_History_5       Family_Hist_2       Family_Hist_3 
# 0.11                0.18                0.42                0.49                0.57 
# Family_Hist_4       Family_Hist_5   Medical_History_1  Medical_History_10  Medical_History_15 
# 0.33                0.70                0.15                0.99                0.75 
# Medical_History_24  Medical_History_32 
# 0.94                0.98 

## check what is the response like for NAs (result: seems random)
table(dt.raw.train$Response[is.na(dt.raw.train$Medical_History_32)])
# 1     2     3     4     5     6     7     8 
# 6059  6413   957  1352  5338 10690  7999 19466
table(dt.raw.train$Response[is.na(dt.raw.train$Medical_History_24)])
# 1     2     3     4     5     6     7     8 
# 5755  6018   928  1327  5078 10047  7580 18847 
table(dt.raw.train$Response[is.na(dt.raw.train$Medical_History_15)])
# 1     2     3     4     5     6     7     8 
# 3981  4518   111   154  4241  8147  6428 17016 
table(dt.raw.train$Response[is.na(dt.raw.train$Medical_History_10)])
# 1     2     3     4     5     6     7     8 
# 6106  6468  1003  1419  5379 11042  7980 19427
table(dt.raw.train$Response[is.na(dt.raw.train$Medical_History_1)])
# 1    2    3    4    5    6    7    8 
# 694  694  231  311  796 1744  998 3421 
table(dt.raw.train$Response[is.na(dt.raw.train$Family_Hist_5)])
# 1     2     3     4     5     6     7     8 
# 3373  4017   745  1119  3701  7835  5221 15800 
table(dt.raw.train$Response[is.na(dt.raw.train$Family_Hist_4)])
# 1    2    3    4    5    6    7    8 
# 3099 2744  303  355 1929 3679 2994 4081 
table(dt.raw.train$Response[is.na(dt.raw.train$Family_Hist_3)])
# 1     2     3     4     5     6     7     8 
# 2709  3101   611   971  3060  6170  4073 13546 
table(dt.raw.train$Response[is.na(dt.raw.train$Family_Hist_2)])
# 1    2    3    4    5    6    7    8 
# 4049 3891  479  551 2840 5628 4354 6864 
table(dt.raw.train$Response[is.na(dt.raw.train$Insurance_History_5)])
# 1    2    3    4    5    6    7    8 
# 2848 2710  565  725 2372 4369 2901 8906 
table(dt.raw.train$Response[is.na(dt.raw.train$Employment_Info_6)])
# 1    2    3    4    5    6    7    8 
# 1171 1194  220  333 1221 1688 1365 3662 
table(dt.raw.train$Response[is.na(dt.raw.train$Employment_Info_4)])
# 1    2    3    4    5    6    7    8 
# 789  679   85  148  560 1752  883 1883
table(dt.raw.train$Response[is.na(dt.raw.train$Employment_Info_1)])
# 1  6 
# 16  3

#######################
## 2.2 class of cols ##
#######################
str(dt.raw.combine)
# nominal
colNominal <- c("Product_Info_1", "Product_Info_2", "Product_Info_3", "Product_Info_5", "Product_Info_6", "Product_Info_7"
                , "Employment_Info_2", "Employment_Info_3", "Employment_Info_5"
                , "InsuredInfo_1", "InsuredInfo_2", "InsuredInfo_3", "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7"
                , "Insurance_History_1", "Insurance_History_2", "Insurance_History_3", "Insurance_History_4", "Insurance_History_7", "Insurance_History_8", "Insurance_History_9"
                , "Family_Hist_1"
                , "Medical_History_2", "Medical_History_3", "Medical_History_4", "Medical_History_5", "Medical_History_6"
                , "Medical_History_7", "Medical_History_8", "Medical_History_9", "Medical_History_10", "Medical_History_11"
                , "Medical_History_12", "Medical_History_13", "Medical_History_14", "Medical_History_16", "Medical_History_17"
                , "Medical_History_18", "Medical_History_19", "Medical_History_20", "Medical_History_21", "Medical_History_22"
                , "Medical_History_23", "Medical_History_25", "Medical_History_26", "Medical_History_27", "Medical_History_28"
                , "Medical_History_29", "Medical_History_30", "Medical_History_31", "Medical_History_33", "Medical_History_34"
                , "Medical_History_35", "Medical_History_36", "Medical_History_37", "Medical_History_38", "Medical_History_39"
                , "Medical_History_40", "Medical_History_41")

# continuous
colContinuous <- c("Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI"
                   , "Employment_Info_1", "Employment_Info_4", "Employment_Info_6"
                   , "Insurance_History_5"
                   , "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5")

# discrete
colDiscrete <- c("Medical_History_1", "Medical_History_15", "Medical_History_24", "Medical_History_32")

######################
## 2.3 unique value ##
######################
# nominal
ColUnique(dt.raw.combine[, colNominal, with = F])
# Product_Info_1      Product_Info_2      Product_Info_3      Product_Info_5      Product_Info_6 
# 2                  19                  38                   2                   2 
# Product_Info_7   Employment_Info_2   Employment_Info_3   Employment_Info_5       InsuredInfo_1 
# 3                  38                   2                   2                   3 
# InsuredInfo_2       InsuredInfo_3       InsuredInfo_4       InsuredInfo_5       InsuredInfo_6 
# 2                  11                   2                   2                   2 
# InsuredInfo_7 Insurance_History_1 Insurance_History_2 Insurance_History_3 Insurance_History_4 
# 2                   2                   3                   3                   3 
# Insurance_History_7 Insurance_History_8 Insurance_History_9       Family_Hist_1   Medical_History_2 
# 3                   3                   3                   3                 628 
# Medical_History_3   Medical_History_4   Medical_History_5   Medical_History_6   Medical_History_7 
# 3                   2                   3                   3                   3 
# Medical_History_8   Medical_History_9  Medical_History_10  Medical_History_11  Medical_History_12 
# 3                   3                 127                   3                   3 
# Medical_History_13  Medical_History_14  Medical_History_16  Medical_History_17  Medical_History_18 
# 3                   3                   3                   3                   3 
# Medical_History_19  Medical_History_20  Medical_History_21  Medical_History_22  Medical_History_23 
# 3                   3                   3                   2                   3 
# Medical_History_25  Medical_History_26  Medical_History_27  Medical_History_28  Medical_History_29 
# 3                   3                   3                   3                   3 
# Medical_History_30  Medical_History_31  Medical_History_33  Medical_History_34  Medical_History_35 
# 3                   3                   3                   3                   3 
# Medical_History_36  Medical_History_37  Medical_History_38  Medical_History_39  Medical_History_40 
# 3                   3                   3                   3                   3 
# Medical_History_41 
# 3 

# discrete
ColUnique(dt.raw.combine[, colDiscrete, with = F])
# Medical_History_1 Medical_History_15 Medical_History_24 Medical_History_32 
# 179                242                234                107

############################################################################################
## 3.0 clean ###############################################################################
############################################################################################
####################################
## 3.1 impute the factor features ##
####################################
colnames.colNAs <- names(ColNAs(dt = dt.raw.combine, method = "sum", output = "nonZero"))
colnames.colNAs
# [1] "Employment_Info_1"   "Employment_Info_4"   "Employment_Info_6"   "Insurance_History_5"
# [5] "Family_Hist_2"       "Family_Hist_3"       "Family_Hist_4"       "Family_Hist_5"      
# [9] "Medical_History_1"   "Medical_History_10"  "Medical_History_15"  "Medical_History_24" 
# [13] "Medical_History_32" 
colNominal








