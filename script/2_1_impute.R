rm(list = ls()); gc();
setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Prudential_Life_Insurance_Assessment/")
load("data/data_clean/dt_raw_combine.RData")
require(data.table)
source("script/utilities/preprocess.R") # utilities functions for preprocessing data
############################################################################################
## 1.0 impute ##############################################################################
############################################################################################
##########################################################
## 1.1 before impute, create new features (Num_of_NAs) ##
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

# add to colDiscrete
colDiscrete <- c(colDiscrete, "Num_of_NAs", "Employment_Info_Num_of_NAs", "Insurance_History_Num_of_NAs"
                 , "Family_Hist_Num_of_NAs", "Medical_History_Num_of_NAs")
dim(dt.raw.combine)
# [1] 79146   134

####################################
## 1.2 impute the factor features ##
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
# character(0)

## impute discrete NAs features
colnames.discrete.NAs <- intersect(colDiscrete, colnames.colNAs)
colnames.discrete.NAs
# [1] "Medical_History_1"  "Medical_History_10" "Medical_History_15" "Medical_History_24"
# [5] "Medical_History_32"

# recall that there are so many NAs in these fields

# Medical_History_1   Medical_History_10    Medical_History_15 
# 0.15                0.99                  0.75 
# Medical_History_24  Medical_History_32 
# 0.94                0.98 

# impute Medical History 10, 24, 32
# dt.raw.combine[, Medical_History_10_Impute_1 := ifelse(is.na(dt.raw.combine$Medical_History_10), -1, dt.raw.combine$Medical_History_10)]
# dt.raw.combine[, Medical_History_24_Impute_1 := ifelse(is.na(dt.raw.combine$Medical_History_24), -1, dt.raw.combine$Medical_History_24)]
# dt.raw.combine[, Medical_History_32_Impute_1 := ifelse(is.na(dt.raw.combine$Medical_History_32), -1, dt.raw.combine$Medical_History_32)]
# colDiscrete <- c(colDiscrete, "Medical_History_10_Impute_1", "Medical_History_24_Impute_1", "Medical_History_32_Impute_1")

# simply remove Medical History 24, 32
dt.raw.combine[, Medical_History_10 := NULL]
dt.raw.combine[, Medical_History_24 := NULL]
dt.raw.combine[, Medical_History_32 := NULL]

# remove from colDiscrete
colDiscrete <- colDiscrete[!colDiscrete %in% c("Medical_History_10", "Medical_History_24", "Medical_History_32")]

# impute Medical History 15
# impute Medical History 15 as median (version 1)
# med.Medical_History_15 <- median(dt.raw.combine$Medical_History_15, na.rm = T)
# dt.raw.combine[, Medical_History_15_Impute_Median := ifelse(is.na(dt.raw.combine$Medical_History_15), med.Medical_History_15, dt.raw.combine$Medical_History_15)]
# impute Medical History 15 as a very large number 2016 (version 2)
dt.raw.combine[, Medical_History_15_Impute_1 := ifelse(is.na(dt.raw.combine$Medical_History_15), -1, dt.raw.combine$Medical_History_15)]
# remove Medical History 15 now
dt.raw.combine[, Medical_History_15 := NULL]
# remove from colDiscrete
colDiscrete <- colDiscrete[colDiscrete != "Medical_History_15"]
# add to colDiscrete
# colDiscrete <- c(colDiscrete, "Medical_History_15_Impute_Median")
colDiscrete <- c(colDiscrete, "Medical_History_15_Impute_1")

# impute Medical History 1
# impute Medical History 1 as median (version 1)
# med.Medical_History_1 <- median(dt.raw.combine$Medical_History_1, na.rm = T)
# dt.raw.combine[, Medical_History_1_Impute_Median := ifelse(is.na(dt.raw.combine$Medical_History_1), med.Medical_History_1, dt.raw.combine$Medical_History_1)]
# impute Medical History 1 as a very large number 2016 (version 2)
dt.raw.combine[, Medical_History_1_Impute_1 := ifelse(is.na(dt.raw.combine$Medical_History_1), -1, dt.raw.combine$Medical_History_1)]
# remove Medical History 1 now
dt.raw.combine[, Medical_History_1 := NULL]
# remove from colDiscrete
colDiscrete <- colDiscrete[colDiscrete != "Medical_History_1"]
# add to colDiscrete
# colDiscrete <- c(colDiscrete, "Medical_History_1_Impute_Median")
colDiscrete <- c(colDiscrete, "Medical_History_1_Impute_1")

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

# add a new feature based on Family_Hist_2 and Family_Hist_3
dt.raw.combine$NewFeature1 <- NA
dt.raw.combine$NewFeature1 <- ifelse(dt.raw.combine$Family_Hist_2>=0 & is.na(dt.raw.combine$Family_Hist_3), 1, dt.raw.combine$NewFeature1)
dt.raw.combine$NewFeature1 <- ifelse(dt.raw.combine$Family_Hist_3>=0 & is.na(dt.raw.combine$Family_Hist_2), 0, dt.raw.combine$NewFeature1)
dt.raw.combine$NewFeature1[is.na(dt.raw.combine$NewFeature1)] <- -1
colNominal <- c("NewFeature1", colNominal)

# # impute as median (version 1)
# median.Employment_Info_1 <- median(dt.raw.combine$Employment_Info_1, na.rm = T)
# median.Employment_Info_4 <- median(dt.raw.combine$Employment_Info_4, na.rm = T)
# median.Employment_Info_6 <- median(dt.raw.combine$Employment_Info_6, na.rm = T)
# median.Insurance_History_5 <- median(dt.raw.combine$Insurance_History_5, na.rm = T)
# median.Family_Hist_2 <- median(dt.raw.combine$Family_Hist_2, na.rm = T)
# median.Family_Hist_3 <- median(dt.raw.combine$Family_Hist_3, na.rm = T)
# median.Family_Hist_4 <- median(dt.raw.combine$Family_Hist_4, na.rm = T)
# median.Family_Hist_5 <- median(dt.raw.combine$Family_Hist_5, na.rm = T)
# 
# dt.raw.combine[, Employment_Info_1_Impute_Median := ifelse(is.na(dt.raw.combine$Employment_Info_1), median.Employment_Info_1, dt.raw.combine$Employment_Info_1)]
# dt.raw.combine[, Employment_Info_4_Impute_Median := ifelse(is.na(dt.raw.combine$Employment_Info_4), median.Employment_Info_4, dt.raw.combine$Employment_Info_4)]
# dt.raw.combine[, Employment_Info_6_Impute_Median := ifelse(is.na(dt.raw.combine$Employment_Info_6), median.Employment_Info_6, dt.raw.combine$Employment_Info_6)]
# dt.raw.combine[, Insurance_History_5_Impute_Median := ifelse(is.na(dt.raw.combine$Insurance_History_5), median.Insurance_History_5, dt.raw.combine$Insurance_History_5)]
# dt.raw.combine[, Family_Hist_2_Impute_Median := ifelse(is.na(dt.raw.combine$Family_Hist_2), median.Family_Hist_2, dt.raw.combine$Family_Hist_2)]
# dt.raw.combine[, Family_Hist_3_Impute_Median := ifelse(is.na(dt.raw.combine$Family_Hist_3), median.Family_Hist_3, dt.raw.combine$Family_Hist_3)]
# dt.raw.combine[, Family_Hist_4_Impute_Median := ifelse(is.na(dt.raw.combine$Family_Hist_4), median.Family_Hist_4, dt.raw.combine$Family_Hist_4)]
# dt.raw.combine[, Family_Hist_5_Impute_Median := ifelse(is.na(dt.raw.combine$Family_Hist_5), median.Family_Hist_5, dt.raw.combine$Family_Hist_5)]
# # add to colContinuous
# colContinuous <- c(colContinuous, "Employment_Info_1_Impute_Median", "Employment_Info_4_Impute_Median", "Employment_Info_6_Impute_Median"
#                    , "Insurance_History_5_Impute_Median", "Family_Hist_2_Impute_Median", "Family_Hist_3_Impute_Median"
#                    , "Family_Hist_4_Impute_Median", "Family_Hist_5_Impute_Median")

# impute as a very large number 1 (version 2)
dt.raw.combine[, Employment_Info_1_Impute_1 := ifelse(is.na(dt.raw.combine$Employment_Info_1), -1, dt.raw.combine$Employment_Info_1)]
dt.raw.combine[, Employment_Info_4_Impute_1 := ifelse(is.na(dt.raw.combine$Employment_Info_4), -1, dt.raw.combine$Employment_Info_4)]
dt.raw.combine[, Employment_Info_6_Impute_1 := ifelse(is.na(dt.raw.combine$Employment_Info_6), -1, dt.raw.combine$Employment_Info_6)]
dt.raw.combine[, Insurance_History_5_Impute_1 := ifelse(is.na(dt.raw.combine$Insurance_History_5), -1, dt.raw.combine$Insurance_History_5)]
dt.raw.combine[, Family_Hist_2_Impute_1 := ifelse(is.na(dt.raw.combine$Family_Hist_2), -1, dt.raw.combine$Family_Hist_2)]
dt.raw.combine[, Family_Hist_3_Impute_1 := ifelse(is.na(dt.raw.combine$Family_Hist_3), -1, dt.raw.combine$Family_Hist_3)]
dt.raw.combine[, Family_Hist_4_Impute_1 := ifelse(is.na(dt.raw.combine$Family_Hist_4), -1, dt.raw.combine$Family_Hist_4)]
dt.raw.combine[, Family_Hist_5_Impute_1 := ifelse(is.na(dt.raw.combine$Family_Hist_5), -1, dt.raw.combine$Family_Hist_5)]
# # add to colContinuous
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

############################################################################################
## 2.0 save ################################################################################
############################################################################################
dt.imputed.combine <- dt.raw.combine
save(dt.imputed.combine, colNominal, colDiscrete, colContinuous, file = "data/data_clean/dt_imputed_combine.RData")