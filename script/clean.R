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
##########################################################
## 3.1 before impute, create a new feature (Num_of_NAs) ##
##########################################################
Num_of_NAs <- apply(dt.raw.combine, 1, function (x) sum(is.na(x)))
Employment_Info_Num_of_NAs <- apply(dt.raw.combine[, grep("Employment_Info", colnames(dt.raw.combine)), with = F], 1, function (x) sum(is.na(x)))
Insurance_History_Num_of_NAs <- apply(dt.raw.combine[, grep("Insurance_History", colnames(dt.raw.combine)), with = F], 1, function (x) sum(is.na(x)))
Family_Hist_Num_of_NAs <- apply(dt.raw.combine[, grep("Family_Hist", colnames(dt.raw.combine)), with = F], 1, function (x) sum(is.na(x)))
Medical_History_Num_of_NAs <- apply(dt.raw.combine[, grep("Medical_History", colnames(dt.raw.combine)), with = F], 1, function (x) sum(is.na(x)))

dt.raw.combine[, Num_of_NAs := Num_of_NAs]
dt.raw.combine[, Employment_Info_Num_of_NAs := Employment_Info_Num_of_NAs]
dt.raw.combine[, Insurance_History_Num_of_NAs := Insurance_History_Num_of_NAs]
dt.raw.combine[, Family_Hist_Num_of_NAs := Family_Hist_Num_of_NAs]
dt.raw.combine[, Medical_History_Num_of_NAs := Medical_History_Num_of_NAs]
dim(dt.raw.combine)
# [1] 79146   134

####################################
## 3.2 impute the factor features ##
####################################
## impute nominal NAs features
colnames.colNAs <- names(ColNAs(dt = dt.raw.combine, method = "sum", output = "nonZero"))
colnames.colNAs
# [1] "Employment_Info_1"   "Employment_Info_4"   "Employment_Info_6"   "Insurance_History_5"
# [5] "Family_Hist_2"       "Family_Hist_3"       "Family_Hist_4"       "Family_Hist_5"      
# [9] "Medical_History_1"   "Medical_History_10"  "Medical_History_15"  "Medical_History_24" 
# [13] "Medical_History_32" 
colNominal
# [1] "Product_Info_1"      "Product_Info_2"      "Product_Info_3"      "Product_Info_5"     
# [5] "Product_Info_6"      "Product_Info_7"      "Employment_Info_2"   "Employment_Info_3"  
# [9] "Employment_Info_5"   "InsuredInfo_1"       "InsuredInfo_2"       "InsuredInfo_3"      
# [13] "InsuredInfo_4"       "InsuredInfo_5"       "InsuredInfo_6"       "InsuredInfo_7"      
# [17] "Insurance_History_1" "Insurance_History_2" "Insurance_History_3" "Insurance_History_4"
# [21] "Insurance_History_7" "Insurance_History_8" "Insurance_History_9" "Family_Hist_1"      
# [25] "Medical_History_2"   "Medical_History_3"   "Medical_History_4"   "Medical_History_5"  
# [29] "Medical_History_6"   "Medical_History_7"   "Medical_History_8"   "Medical_History_9"  
# [33] "Medical_History_10"  "Medical_History_11"  "Medical_History_12"  "Medical_History_13" 
# [37] "Medical_History_14"  "Medical_History_16"  "Medical_History_17"  "Medical_History_18" 
# [41] "Medical_History_19"  "Medical_History_20"  "Medical_History_21"  "Medical_History_22" 
# [45] "Medical_History_23"  "Medical_History_25"  "Medical_History_26"  "Medical_History_27" 
# [49] "Medical_History_28"  "Medical_History_29"  "Medical_History_30"  "Medical_History_31" 
# [53] "Medical_History_33"  "Medical_History_34"  "Medical_History_35"  "Medical_History_36" 
# [57] "Medical_History_37"  "Medical_History_38"  "Medical_History_39"  "Medical_History_40" 
# [61] "Medical_History_41" 

colnames.nominal.NAs <- intersect(colNominal, colnames.colNAs)
colnames.nominal.NAs
# [1] "Medical_History_10"

# recall that there are so many NAs in this field
# Medical_History_10
# 0.99

# simply remove it
dt.raw.combine[, Medical_History_10 := NULL]
# remove from colNominal
colNominal <- colNominal[colNominal != "Medical_History_10"]

## impute discrete NAs features
colnames.discrete.NAs <- intersect(colDiscrete, colnames.colNAs)
colnames.discrete.NAs
# [1] "Medical_History_1"  "Medical_History_15" "Medical_History_24" "Medical_History_32"

# recall that there are so many NAs in these fields

# Medical_History_1   Medical_History_15 
# 0.15                0.75 
# Medical_History_24  Medical_History_32 
# 0.94                0.98 

# impute Medical History 24, 32
# simply remove Medical History 24, 32
dt.raw.combine[, Medical_History_24 := NULL]
dt.raw.combine[, Medical_History_32 := NULL]
# remove from colDiscrete
colDiscrete <- colDiscrete[!colDiscrete %in% c("Medical_History_24", "Medical_History_32")]

# impute Medical History 15
# impute Medical History 15 as median (version 1)
med.Medical_History_15 <- median(dt.raw.combine$Medical_History_15, na.rm = T)
dt.raw.combine[, Medical_History_15_Impute_Median := ifelse(is.na(dt.raw.combine$Medical_History_15), med.Medical_History_15, dt.raw.combine$Medical_History_15)]
# impute Medical History 15 as a very large number 2016 (version 2)
dt.raw.combine[, Medical_History_15_Impute_2016 := ifelse(is.na(dt.raw.combine$Medical_History_15), 2016, dt.raw.combine$Medical_History_15)]
# remove Medical History 15 now
dt.raw.combine[, Medical_History_15 := NULL]
# remove from colDiscrete
colDiscrete <- colDiscrete[colDiscrete != "Medical_History_15"]
# add to colDiscrete
colDiscrete <- c(colDiscrete, "Medical_History_15_Impute_Median", "Medical_History_15_Impute_2016")

# impute Medical History 1
# impute Medical History 1 as median (version 1)
med.Medical_History_1 <- median(dt.raw.combine$Medical_History_1, na.rm = T)
dt.raw.combine[, Medical_History_1_Impute_Median := ifelse(is.na(dt.raw.combine$Medical_History_1), med.Medical_History_1, dt.raw.combine$Medical_History_1)]
# impute Medical History 1 as a very large number 2016 (version 2)
dt.raw.combine[, Medical_History_1_Impute_2016 := ifelse(is.na(dt.raw.combine$Medical_History_1), 2016, dt.raw.combine$Medical_History_1)]
# remove Medical History 1 now
dt.raw.combine[, Medical_History_1 := NULL]
# remove from colDiscrete
colDiscrete <- colDiscrete[colDiscrete != "Medical_History_1"]
# add to colDiscrete
colDiscrete <- c(colDiscrete, "Medical_History_1_Impute_Median", "Medical_History_1_Impute_2016")

## impute continual NAs features
colnames.continuous.NAs <- intersect(colContinuous, colnames.colNAs)
colnames.continuous.NAs
# [1] "Employment_Info_1"   "Employment_Info_4"   "Employment_Info_6"   "Insurance_History_5"
# [5] "Family_Hist_2"       "Family_Hist_3"       "Family_Hist_4"       "Family_Hist_5"

# proportion of NAs
# Employment_Info_4   Employment_Info_6 Insurance_History_5       Family_Hist_2       Family_Hist_3 
# 0.11                0.18                0.42                0.49                0.57 
# Family_Hist_4       Family_Hist_5
# 0.33                0.70

# impute as mean (version 1)
mean.Employment_Info_1 <- mean(dt.raw.combine$Employment_Info_1, na.rm = T)
mean.Employment_Info_4 <- mean(dt.raw.combine$Employment_Info_2, na.rm = T)
mean.Employment_Info_6 <- mean(dt.raw.combine$Employment_Info_6, na.rm = T)
mean.Insurance_History_5 <- mean(dt.raw.combine$Insurance_History_5, na.rm = T)
mean.Family_Hist_2 <- mean(dt.raw.combine$Family_Hist_2, na.rm = T)
mean.Family_Hist_3 <- mean(dt.raw.combine$Family_Hist_3, na.rm = T)
mean.Family_Hist_4 <- mean(dt.raw.combine$Family_Hist_4, na.rm = T)
mean.Family_Hist_5 <- mean(dt.raw.combine$Family_Hist_5, na.rm = T)

dt.raw.combine[, Employment_Info_1_Impute_Mean := ifelse(is.na(dt.raw.combine$Employment_Info_1), mean.Employment_Info_1, dt.raw.combine$Employment_Info_1)]
dt.raw.combine[, Employment_Info_4_Impute_Mean := ifelse(is.na(dt.raw.combine$Employment_Info_4), mean.Employment_Info_4, dt.raw.combine$Employment_Info_4)]
dt.raw.combine[, Employment_Info_6_Impute_Mean := ifelse(is.na(dt.raw.combine$Employment_Info_6), mean.Employment_Info_6, dt.raw.combine$Employment_Info_6)]
dt.raw.combine[, Insurance_History_5_Impute_Mean := ifelse(is.na(dt.raw.combine$Insurance_History_5), mean.Insurance_History_5, dt.raw.combine$Insurance_History_5)]
dt.raw.combine[, Family_Hist_2_Impute_Mean := ifelse(is.na(dt.raw.combine$Family_Hist_2), mean.Family_Hist_2, dt.raw.combine$Family_Hist_2)]
dt.raw.combine[, Family_Hist_3_Impute_Mean := ifelse(is.na(dt.raw.combine$Family_Hist_3), mean.Family_Hist_3, dt.raw.combine$Family_Hist_3)]
dt.raw.combine[, Family_Hist_4_Impute_Mean := ifelse(is.na(dt.raw.combine$Family_Hist_4), mean.Family_Hist_4, dt.raw.combine$Family_Hist_4)]
dt.raw.combine[, Family_Hist_5_Impute_Mean := ifelse(is.na(dt.raw.combine$Family_Hist_5), mean.Family_Hist_5, dt.raw.combine$Family_Hist_5)]
# add to colContinuous
colContinuous <- c(colContinuous, "Employment_Info_1_Impute_Mean", "Employment_Info_4_Impute_Mean", "Employment_Info_6_Impute_Mean"
                   , "Insurance_History_5_Impute_Mean", "Family_Hist_2_Impute_Mean", "Family_Hist_3_Impute_Mean"
                   , "Family_Hist_4_Impute_Mean", "Family_Hist_5_Impute_Mean")

# impute as a very large number 1
dt.raw.combine[, Employment_Info_1_Impute_1 := ifelse(is.na(dt.raw.combine$Employment_Info_1), 1, dt.raw.combine$Employment_Info_1)]
dt.raw.combine[, Employment_Info_4_Impute_1 := ifelse(is.na(dt.raw.combine$Employment_Info_4), 1, dt.raw.combine$Employment_Info_4)]
dt.raw.combine[, Employment_Info_6_Impute_1 := ifelse(is.na(dt.raw.combine$Employment_Info_6), 1, dt.raw.combine$Employment_Info_6)]
dt.raw.combine[, Insurance_History_5_Impute_1 := ifelse(is.na(dt.raw.combine$Insurance_History_5), 1, dt.raw.combine$Insurance_History_5)]
dt.raw.combine[, Family_Hist_2_Impute_1 := ifelse(is.na(dt.raw.combine$Family_Hist_2), 1, dt.raw.combine$Family_Hist_2)]
dt.raw.combine[, Family_Hist_3_Impute_1 := ifelse(is.na(dt.raw.combine$Family_Hist_3), 1, dt.raw.combine$Family_Hist_3)]
dt.raw.combine[, Family_Hist_4_Impute_1 := ifelse(is.na(dt.raw.combine$Family_Hist_4), 1, dt.raw.combine$Family_Hist_4)]
dt.raw.combine[, Family_Hist_5_Impute_1 := ifelse(is.na(dt.raw.combine$Family_Hist_5), 1, dt.raw.combine$Family_Hist_5)]
# add to colContinuous
colContinuous <- c(colContinuous, "Employment_Info_1_Impute_1", "Employment_Info_4_Impute_1", "Employment_Info_6_Impute_1"
                   , "Insurance_History_5_Impute_1", "Family_Hist_2_Impute_1", "Family_Hist_3_Impute_1"
                   , "Family_Hist_4_Impute_1", "Family_Hist_5_Impute_1")

# remove original features
dt.raw.combine[, Employment_Info_1 := NULL]
dt.raw.combine[, Employment_Info_4 := NULL]
dt.raw.combine[, Employment_Info_6 := NULL]
dt.raw.combine[, Insurance_History_5 := NULL]
dt.raw.combine[, Family_Hist_2 := NULL]
dt.raw.combine[, Family_Hist_3 := NULL]
dt.raw.combine[, Family_Hist_4 := NULL]
dt.raw.combine[, Family_Hist_5 := NULL]
# remove from colContinuous
colContinuous <- colContinuous[!colContinuous %in% c("Employment_Info_1", "Employment_Info_4", "Employment_Info_6"
                                                     , "Insurance_History_5", "Family_Hist_2", "Family_Hist_3"
                                                     , "Family_Hist_4", "Family_Hist_5")]

# check again on NAs
ColNAs(dt.raw.combine, method = "sum", output = "NonZero")
# [1] FALSE # cool all imputed!

#######################################
## 3.3 sort out the class of columns ##
#######################################
###################
## 3.3.1 nominal ##
###################
colNominal
# no. of levels 
no.of.levels <- sapply(dt.raw.combine[, colNominal, with = F], function (x) {length(names(table(x)))})
###############################
## 3.3.1.1 no. of levels > 3 ##
###############################
no.of.levels[no.of.levels > 3]
# Product_Info_2    Product_Info_3 Employment_Info_2     InsuredInfo_3 Medical_History_2 
# 19                38                38                11               628 
colNominal.needBinEnc <- names(no.of.levels[no.of.levels > 3])

# binary encoding for no. of levels > 3
# before that, create a new feature for Product_Info_2
sort(unique(dt.raw.combine$Product_Info_2))
# [1] "A1" "A2" "A3" "A4" "A5" "A6" "A7" "A8" "B1" "B2" "C1" "C2" "C3" "C4" "D1" "D2" "D3" "D4" "E1"

dt.raw.combine[, Product_Info_2_A := ifelse(grepl("A", dt.raw.combine$Product_Info_2), 1, 0)]
dt.raw.combine[, Product_Info_2_B := ifelse(grepl("B", dt.raw.combine$Product_Info_2), 1, 0)]
dt.raw.combine[, Product_Info_2_C := ifelse(grepl("C", dt.raw.combine$Product_Info_2), 1, 0)]
dt.raw.combine[, Product_Info_2_D := ifelse(grepl("D", dt.raw.combine$Product_Info_2), 1, 0)]
dt.raw.combine[, Product_Info_2_E := ifelse(grepl("E", dt.raw.combine$Product_Info_2), 1, 0)]

dt.raw.combine[, Product_Info_2_1 := ifelse(grepl("1", dt.raw.combine$Product_Info_2), 1, 0)]

# now start handling the no. of levels > 3
dt.raw.combine <- BinaryEncode(dt.raw.combine, colNominal.needBinEnc)
# remove colNominal.needBinEnc from colNominal
colNominal <- colNominal[!colNominal %in% colNominal.needBinEnc]

########################################
## 3.3.1.2 no. of levels == 2 ##
########################################
colNominal.needOnehot <- names(no.of.levels[no.of.levels == 2])
colNominal.needOnehot
# [1] "Product_Info_1"      "Product_Info_5"      "Product_Info_6"      "Employment_Info_3"  
# [5] "Employment_Info_5"   "InsuredInfo_2"       "InsuredInfo_4"       "InsuredInfo_5"      
# [9] "InsuredInfo_6"       "InsuredInfo_7"       "Insurance_History_1" "Medical_History_4"  
# [13] "Medical_History_22" 
# what are the values
# before
sapply(dt.raw.combine[, colNominal.needOnehot, with = F], unique) # they are all 1 and 2.
#       Product_Info_1 Product_Info_5 Product_Info_6 Employment_Info_3 Employment_Info_5 InsuredInfo_2
# [1,]              0              2              1                 1                 3             2
# [2,]              1              3              3                 3                 2             3
#       InsuredInfo_4 InsuredInfo_5 InsuredInfo_6 InsuredInfo_7 Insurance_History_1 Medical_History_4
# [1,]             3             1             2             1                   1                 1
# [2,]             2             3             1             3                   2                 2
#       Medical_History_22
# [1,]                  2
# [2,]                  1
# OneHot (a simple one), changing 1 and 2 to 0 and 1
dt.raw.combine[, Product_Info_1 := Product_Info_1 - 1]
dt.raw.combine[, Product_Info_5 := Product_Info_5 - 2]
dt.raw.combine[, Product_Info_6 := ifelse(dt.raw.combine$Product_Info_6 == 1, 0, 1)]
dt.raw.combine[, Employment_Info_3 := ifelse(dt.raw.combine$Employment_Info_3 == 1, 0, 1)]
dt.raw.combine[, Employment_Info_5 := Employment_Info_5 - 2]
dt.raw.combine[, InsuredInfo_2 := InsuredInfo_2 - 2]
dt.raw.combine[, InsuredInfo_4 := InsuredInfo_4 - 2]
dt.raw.combine[, InsuredInfo_5 := ifelse(dt.raw.combine$InsuredInfo_5 == 1, 0, 1)]
dt.raw.combine[, InsuredInfo_6 := InsuredInfo_6 - 1]
dt.raw.combine[, InsuredInfo_7 := ifelse(dt.raw.combine$InsuredInfo_7 == 1, 0, 1)]
dt.raw.combine[, Insurance_History_1 := Insurance_History_1 - 1]
dt.raw.combine[, Medical_History_4 := Medical_History_4 - 1]
dt.raw.combine[, Medical_History_22 := Medical_History_22 - 1]
# after
sapply(dt.raw.combine[, colNominal.needOnehot, with = F], unique)
# Product_Info_1 Product_Info_5 Product_Info_6 Employment_Info_3 Employment_Info_5 InsuredInfo_2
# [1,]              0              0              0                 0                 1             0
# [2,]              1              1              1                 1                 0             1
# InsuredInfo_4 InsuredInfo_5 InsuredInfo_6 InsuredInfo_7 Insurance_History_1 Medical_History_4
# [1,]             1             0             1             0                   0                 0
# [2,]             0             1             0             1                   1                 1
# Medical_History_22
# [1,]                  1
# [2,]                  0
# class
sapply(dt.raw.combine[, colNominal.needOnehot, with = F], class)
# Product_Info_1      Product_Info_5      Product_Info_6   Employment_Info_3   Employment_Info_5 
# "numeric"           "numeric"           "numeric"           "numeric"           "numeric" 
# InsuredInfo_2       InsuredInfo_4       InsuredInfo_5       InsuredInfo_6       InsuredInfo_7 
# "numeric"           "numeric"           "numeric"           "numeric"           "numeric" 
# Insurance_History_1   Medical_History_4  Medical_History_22 
# "numeric"           "numeric"           "numeric"


########################################
## 3.3.1.3 no. of levels == 3 ##########
########################################
# let it be, later we will use caret.dummyVars to create dummy variable
# but change them to factor vars
colNominal.needDummyVars <- names(no.of.levels[no.of.levels == 3])
colNominal.needDummyVars
# [1] "Product_Info_7"      "InsuredInfo_1"       "Insurance_History_2" "Insurance_History_3"
# [5] "Insurance_History_4" "Insurance_History_7" "Insurance_History_8" "Insurance_History_9"
# [9] "Family_Hist_1"       "Medical_History_3"   "Medical_History_5"   "Medical_History_6"  
# [13] "Medical_History_7"   "Medical_History_8"   "Medical_History_9"   "Medical_History_11" 
# [17] "Medical_History_12"  "Medical_History_13"  "Medical_History_14"  "Medical_History_16" 
# [21] "Medical_History_17"  "Medical_History_18"  "Medical_History_19"  "Medical_History_20" 
# [25] "Medical_History_21"  "Medical_History_23"  "Medical_History_25"  "Medical_History_26" 
# [29] "Medical_History_27"  "Medical_History_28"  "Medical_History_29"  "Medical_History_30" 
# [33] "Medical_History_31"  "Medical_History_33"  "Medical_History_34"  "Medical_History_35" 
# [37] "Medical_History_36"  "Medical_History_37"  "Medical_History_38"  "Medical_History_39" 
# [41] "Medical_History_40"  "Medical_History_41" 
dt.raw.combine.needDummyVars <- dt.raw.combine[, c("Id", colNominal.needDummyVars), with = F][, lapply(.SD, as.factor), by = Id]
dt.raw.combine <- data.table(dt.raw.combine[, !c("Id", colNominal.needDummyVars), with = F], dt.raw.combine.needDummyVars)
# [1] 79146   173

####################
## 3.3.2 discrete ##
####################
dt.raw.combine[, colDiscrete, with = F]
sapply(dt.raw.combine[, colDiscrete, with = F], class)
# Medical_History_15_Impute_Median   Medical_History_15_Impute_2016  Medical_History_1_Impute_Median 
# "numeric"                        "numeric"                        "integer" 
# Medical_History_1_Impute_2016 
# "numeric"

# all good

######################
## 3.3.3 continuous ##
######################
dt.raw.combine[, colContinuous, with = F]
sapply(dt.raw.combine[, colContinuous, with = F], class)
# Product_Info_4                         Ins_Age                              Ht 
# "numeric"                       "numeric"                       "numeric" 
# Wt                             BMI   Employment_Info_1_Impute_Mean 
# "numeric"                       "numeric"                       "numeric" 
# Employment_Info_4_Impute_Mean   Employment_Info_6_Impute_Mean Insurance_History_5_Impute_Mean 
# "numeric"                       "numeric"                       "numeric" 
# Family_Hist_2_Impute_Mean       Family_Hist_3_Impute_Mean       Family_Hist_4_Impute_Mean 
# "numeric"                       "numeric"                       "numeric" 
# Family_Hist_5_Impute_Mean      Employment_Info_1_Impute_1      Employment_Info_4_Impute_1 
# "numeric"                       "numeric"                       "numeric" 
# Employment_Info_6_Impute_1    Insurance_History_5_Impute_1          Family_Hist_2_Impute_1 
# "numeric"                       "numeric"                       "numeric" 
# Family_Hist_3_Impute_1          Family_Hist_4_Impute_1          Family_Hist_5_Impute_1 
# "numeric"                       "numeric"                       "numeric" 

# all good

## stats of dt.raw.combine after imputation and class-ify
dim(dt.raw.combine)
# [1] 79146   173
sort(names(dt.raw.combine))
# [1] "BMI"                              "Employment_Info_1_Impute_1"      
# [3] "Employment_Info_1_Impute_Mean"    "Employment_Info_2_bin_1"         
# [5] "Employment_Info_2_bin_2"          "Employment_Info_2_bin_3"         
# [7] "Employment_Info_2_bin_4"          "Employment_Info_2_bin_5"         
# [9] "Employment_Info_2_bin_6"          "Employment_Info_3"               
# [11] "Employment_Info_4_Impute_1"       "Employment_Info_4_Impute_Mean"   
# [13] "Employment_Info_5"                "Employment_Info_6_Impute_1"      
# [15] "Employment_Info_6_Impute_Mean"    "Employment_Info_Num_of_NAs"      
# [17] "Family_Hist_1"                    "Family_Hist_2_Impute_1"          
# [19] "Family_Hist_2_Impute_Mean"        "Family_Hist_3_Impute_1"          
# [21] "Family_Hist_3_Impute_Mean"        "Family_Hist_4_Impute_1"          
# [23] "Family_Hist_4_Impute_Mean"        "Family_Hist_5_Impute_1"          
# [25] "Family_Hist_5_Impute_Mean"        "Family_Hist_Num_of_NAs"          
# [27] "Ht"                               "Id"                              
# [29] "Ins_Age"                          "Insurance_History_1"             
# [31] "Insurance_History_2"              "Insurance_History_3"             
# [33] "Insurance_History_4"              "Insurance_History_5_Impute_1"    
# [35] "Insurance_History_5_Impute_Mean"  "Insurance_History_7"             
# [37] "Insurance_History_8"              "Insurance_History_9"             
# [39] "Insurance_History_Num_of_NAs"     "InsuredInfo_1"                   
# [41] "InsuredInfo_2"                    "InsuredInfo_3_bin_1"             
# [43] "InsuredInfo_3_bin_2"              "InsuredInfo_3_bin_3"             
# [45] "InsuredInfo_3_bin_4"              "InsuredInfo_4"                   
# [47] "InsuredInfo_5"                    "InsuredInfo_6"                   
# [49] "InsuredInfo_7"                    "isTest"                          
# [51] "Medical_History_1_Impute_2016"    "Medical_History_1_Impute_Median" 
# [53] "Medical_History_11"               "Medical_History_12"              
# [55] "Medical_History_13"               "Medical_History_14"              
# [57] "Medical_History_15_Impute_2016"   "Medical_History_15_Impute_Median"
# [59] "Medical_History_16"               "Medical_History_17"              
# [61] "Medical_History_18"               "Medical_History_19"              
# [63] "Medical_History_2_bin_1"          "Medical_History_2_bin_10"        
# [65] "Medical_History_2_bin_2"          "Medical_History_2_bin_3"         
# [67] "Medical_History_2_bin_4"          "Medical_History_2_bin_5"         
# [69] "Medical_History_2_bin_6"          "Medical_History_2_bin_7"         
# [71] "Medical_History_2_bin_8"          "Medical_History_2_bin_9"         
# [73] "Medical_History_20"               "Medical_History_21"              
# [75] "Medical_History_22"               "Medical_History_23"              
# [77] "Medical_History_25"               "Medical_History_26"              
# [79] "Medical_History_27"               "Medical_History_28"              
# [81] "Medical_History_29"               "Medical_History_3"               
# [83] "Medical_History_30"               "Medical_History_31"              
# [85] "Medical_History_33"               "Medical_History_34"              
# [87] "Medical_History_35"               "Medical_History_36"              
# [89] "Medical_History_37"               "Medical_History_38"              
# [91] "Medical_History_39"               "Medical_History_4"               
# [93] "Medical_History_40"               "Medical_History_41"              
# [95] "Medical_History_5"                "Medical_History_6"               
# [97] "Medical_History_7"                "Medical_History_8"               
# [99] "Medical_History_9"                "Medical_History_Num_of_NAs"      
# [101] "Medical_Keyword_1"                "Medical_Keyword_10"              
# [103] "Medical_Keyword_11"               "Medical_Keyword_12"              
# [105] "Medical_Keyword_13"               "Medical_Keyword_14"              
# [107] "Medical_Keyword_15"               "Medical_Keyword_16"              
# [109] "Medical_Keyword_17"               "Medical_Keyword_18"              
# [111] "Medical_Keyword_19"               "Medical_Keyword_2"               
# [113] "Medical_Keyword_20"               "Medical_Keyword_21"              
# [115] "Medical_Keyword_22"               "Medical_Keyword_23"              
# [117] "Medical_Keyword_24"               "Medical_Keyword_25"              
# [119] "Medical_Keyword_26"               "Medical_Keyword_27"              
# [121] "Medical_Keyword_28"               "Medical_Keyword_29"              
# [123] "Medical_Keyword_3"                "Medical_Keyword_30"              
# [125] "Medical_Keyword_31"               "Medical_Keyword_32"              
# [127] "Medical_Keyword_33"               "Medical_Keyword_34"              
# [129] "Medical_Keyword_35"               "Medical_Keyword_36"              
# [131] "Medical_Keyword_37"               "Medical_Keyword_38"              
# [133] "Medical_Keyword_39"               "Medical_Keyword_4"               
# [135] "Medical_Keyword_40"               "Medical_Keyword_41"              
# [137] "Medical_Keyword_42"               "Medical_Keyword_43"              
# [139] "Medical_Keyword_44"               "Medical_Keyword_45"              
# [141] "Medical_Keyword_46"               "Medical_Keyword_47"              
# [143] "Medical_Keyword_48"               "Medical_Keyword_5"               
# [145] "Medical_Keyword_6"                "Medical_Keyword_7"               
# [147] "Medical_Keyword_8"                "Medical_Keyword_9"               
# [149] "Num_of_NAs"                       "Product_Info_1"                  
# [151] "Product_Info_2_1"                 "Product_Info_2_A"                
# [153] "Product_Info_2_B"                 "Product_Info_2_bin_1"            
# [155] "Product_Info_2_bin_2"             "Product_Info_2_bin_3"            
# [157] "Product_Info_2_bin_4"             "Product_Info_2_bin_5"            
# [159] "Product_Info_2_C"                 "Product_Info_2_D"                
# [161] "Product_Info_2_E"                 "Product_Info_3_bin_1"            
# [163] "Product_Info_3_bin_2"             "Product_Info_3_bin_3"            
# [165] "Product_Info_3_bin_4"             "Product_Info_3_bin_5"            
# [167] "Product_Info_3_bin_6"             "Product_Info_4"                  
# [169] "Product_Info_5"                   "Product_Info_6"                  
# [171] "Product_Info_7"                   "Response"                        
# [173] "Wt"  

############################################################################################
## 3.0 save ################################################################################
############################################################################################
dt.clean.combine <- dt.raw.combine
save(dt.clean.combine, file = "data/data_clean/dt_clean_combine.RData")







