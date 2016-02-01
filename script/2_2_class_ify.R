rm(list = ls()); gc();
setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Prudential_Life_Insurance_Assessment/")
load("data/data_clean/dt_imputed_combine.RData")
require(data.table)
source("script/utilities/preprocess.R") # utilities functions for preprocessing data
############################################################################################
## 1.0 class-ify ###########################################################################
############################################################################################
#################
## 1.1 nominal ##
#################
colNominal
# no. of levels 
no.of.levels <- sapply(dt.imputed.combine[, colNominal, with = F], function (x) {length(names(table(x)))})
#############################
## 1.1.1 no. of levels > 3 ##
#############################
no.of.levels[no.of.levels > 3]
# Product_Info_2    Product_Info_3 Employment_Info_2     InsuredInfo_3 Medical_History_2 
# 19                38                38                11               628 
colNominal.needBinEnc <- names(no.of.levels[no.of.levels > 3])

# binary encoding for no. of levels > 3
# before that, create a new feature for Product_Info_2
sort(unique(dt.imputed.combine$Product_Info_2))
# [1] "A1" "A2" "A3" "A4" "A5" "A6" "A7" "A8" "B1" "B2" "C1" "C2" "C3" "C4" "D1" "D2" "D3" "D4" "E1"

dt.imputed.combine[, Product_Info_2_A := ifelse(grepl("A", dt.imputed.combine$Product_Info_2), 1, 0)]
dt.imputed.combine[, Product_Info_2_B := ifelse(grepl("B", dt.imputed.combine$Product_Info_2), 1, 0)]
dt.imputed.combine[, Product_Info_2_C := ifelse(grepl("C", dt.imputed.combine$Product_Info_2), 1, 0)]
dt.imputed.combine[, Product_Info_2_D := ifelse(grepl("D", dt.imputed.combine$Product_Info_2), 1, 0)]
dt.imputed.combine[, Product_Info_2_E := ifelse(grepl("E", dt.imputed.combine$Product_Info_2), 1, 0)]

dt.imputed.combine[, Product_Info_2_1 := ifelse(grepl("1", dt.imputed.combine$Product_Info_2), 1, 0)]

# now start handling the no. of levels > 3
dt.imputed.combine <- ConvertNonNumFactorToNumFactor(dt.imputed.combine, "Product_Info_2")
colNominal <- c(colNominal, "Product_Info_2_toNum")
colNominal.needBinEnc <- c(colNominal.needBinEnc, "Product_Info_2_toNum")
colNominal.needBinEnc <- colNominal.needBinEnc[colNominal.needBinEnc != "Product_Info_2"]

# dt.imputed.combine <- BinaryEncode(dt.imputed.combine, colNominal.needBinEnc)

# remove Product_Info_2
dt.imputed.combine[, Product_Info_2 := NULL]
colNominal <- colNominal[colNominal != "Product_Info_2"]
# # add to colNominal
# colNominal.newBinEnc <- as.character()
# # binary encoded cols
# for (col in colNominal.needBinEnc){
#     col <- paste(col, "_bin", sep = "")
#     colNominal.newBinEnc <- c(colNominal.newBinEnc, names(dt.imputed.combine)[grep(col, names(dt.imputed.combine))])
# }
# # non-numeric factors to numeric factors
# for (col in colNominal.needBinEnc){
#     col <- paste(col, "_toNum", sep = "")
#     colNominal.newBinEnc <- c(colNominal.newBinEnc, names(dt.imputed.combine)[grep(col, names(dt.imputed.combine))])
# }
# colNominal <- c(colNominal, colNominal.newBinEnc)
# # remove colNominal.needBinEnc from colNominal
# colNominal <- colNominal[!colNominal %in% colNominal.needBinEnc]
# add to colNominal
colNominal <- c(colNominal, "Product_Info_2_A", "Product_Info_2_B", "Product_Info_2_C", "Product_Info_2_D"
                , "Product_Info_2_E", "Product_Info_2_1")

##############################
## 1.1.2 no. of levels == 2 ##
##############################
colNominal.needOnehot <- names(no.of.levels[no.of.levels == 2])
colNominal.needOnehot
# [1] "Product_Info_1"      "Product_Info_5"      "Product_Info_6"      "Employment_Info_3"  
# [5] "Employment_Info_5"   "InsuredInfo_2"       "InsuredInfo_4"       "InsuredInfo_5"      
# [9] "InsuredInfo_6"       "InsuredInfo_7"       "Insurance_History_1" "Medical_History_4"  
# [13] "Medical_History_22" 
# what are the values
# before
sapply(dt.imputed.combine[, colNominal.needOnehot, with = F], unique) # they are all 1 and 2.
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
dt.imputed.combine[, Product_Info_1 := Product_Info_1 - 1]
dt.imputed.combine[, Product_Info_5 := Product_Info_5 - 2]
dt.imputed.combine[, Product_Info_6 := ifelse(dt.imputed.combine$Product_Info_6 == 1, 0, 1)]
dt.imputed.combine[, Employment_Info_3 := ifelse(dt.imputed.combine$Employment_Info_3 == 1, 0, 1)]
dt.imputed.combine[, Employment_Info_5 := Employment_Info_5 - 2]
dt.imputed.combine[, InsuredInfo_2 := InsuredInfo_2 - 2]
dt.imputed.combine[, InsuredInfo_4 := InsuredInfo_4 - 2]
dt.imputed.combine[, InsuredInfo_5 := ifelse(dt.imputed.combine$InsuredInfo_5 == 1, 0, 1)]
dt.imputed.combine[, InsuredInfo_6 := InsuredInfo_6 - 1]
dt.imputed.combine[, InsuredInfo_7 := ifelse(dt.imputed.combine$InsuredInfo_7 == 1, 0, 1)]
dt.imputed.combine[, Insurance_History_1 := Insurance_History_1 - 1]
dt.imputed.combine[, Medical_History_4 := Medical_History_4 - 1]
dt.imputed.combine[, Medical_History_22 := Medical_History_22 - 1]
# after
sapply(dt.imputed.combine[, colNominal.needOnehot, with = F], unique)
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
sapply(dt.imputed.combine[, colNominal.needOnehot, with = F], class)
# Product_Info_1      Product_Info_5      Product_Info_6   Employment_Info_3   Employment_Info_5 
# "numeric"           "numeric"           "numeric"           "numeric"           "numeric" 
# InsuredInfo_2       InsuredInfo_4       InsuredInfo_5       InsuredInfo_6       InsuredInfo_7 
# "numeric"           "numeric"           "numeric"           "numeric"           "numeric" 
# Insurance_History_1   Medical_History_4  Medical_History_22 
# "numeric"           "numeric"           "numeric"


########################################
## 1.1.3 no. of levels == 3 ############
########################################
# let it be, later we will use caret.dummyVars to create dummy variable
# but change them to factor vars
colNominal.needDummyVars <- names(no.of.levels[no.of.levels == 3])
colNominal.needDummyVars
# [1] "NewFeature1"         "Product_Info_7"      "InsuredInfo_1"       "Insurance_History_2" "Insurance_History_3"
# [6] "Insurance_History_4" "Insurance_History_7" "Insurance_History_8" "Insurance_History_9" "Family_Hist_1"      
# [11] "Medical_History_3"   "Medical_History_5"   "Medical_History_6"   "Medical_History_7"   "Medical_History_8"  
# [16] "Medical_History_9"   "Medical_History_11"  "Medical_History_12"  "Medical_History_13"  "Medical_History_14" 
# [21] "Medical_History_16"  "Medical_History_17"  "Medical_History_18"  "Medical_History_19"  "Medical_History_20" 
# [26] "Medical_History_21"  "Medical_History_23"  "Medical_History_25"  "Medical_History_26"  "Medical_History_27" 
# [31] "Medical_History_28"  "Medical_History_29"  "Medical_History_30"  "Medical_History_31"  "Medical_History_33" 
# [36] "Medical_History_34"  "Medical_History_35"  "Medical_History_36"  "Medical_History_37"  "Medical_History_38" 
# [41] "Medical_History_39"  "Medical_History_40"  "Medical_History_41" 

##################
## 1.2 discrete ##
##################
dt.imputed.combine[, colDiscrete, with = F]
sapply(dt.imputed.combine[, colDiscrete, with = F], class)
# Medical_History_15_Impute_Median   Medical_History_15_Impute_2016  Medical_History_1_Impute_Median 
# "numeric"                        "numeric"                        "integer" 
# Medical_History_1_Impute_2016 
# "numeric"

# all good

####################
## 1.3 continuous ##
####################
dt.imputed.combine[, colContinuous, with = F]
sapply(dt.imputed.combine[, colContinuous, with = F], class)
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

## stats of dt.imputed.combine after imputation and class-ify
dim(dt.imputed.combine)
# [1] 79146   173
sort(names(dt.imputed.combine))
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

###################
## 1.4 class-ify ##
###################
# colNomial to factor
# previously forgot the medical_keyword, now add them in
colNominal <- c(colNominal, names(dt.imputed.combine)[grepl("Medical_Keyword", names(dt.imputed.combine))])
dt.imputed.combine.nominal <- dt.imputed.combine[, colNominal, with = F][, lapply(.SD, as.factor)]
dt.imputed.combine <- data.table(dt.imputed.combine[, !colNominal, with = F], dt.imputed.combine.nominal)

############################################################################################
## 2.0 save ################################################################################
############################################################################################
dt.class.ified.combine <- dt.imputed.combine
save(dt.class.ified.combine, colNominal, colDiscrete, colContinuous, file = "data/data_clean/dt_class_ified_combine.RData")







