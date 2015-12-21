rm(list = ls()); gc();
setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Prudential_Life_Insurance_Assessment/")
load("data/data_clean/dt_class_ified_combine.RData")
require(data.table)
require(caret)
############################################################################################
## 1.0 preprocess ##########################################################################
############################################################################################
####################
## 1.1 dummyVsars ##
####################
# levels == 3
no.of.levels <- sapply(dt.class.ified.combine[, colNominal, with = F], function (x) {length(names(table(x)))})
colnames.levels3 <- names(no.of.levels)[no.of.levels == 3]
colnames.levels3
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

formula.levels3 <- paste("~", paste(colnames.levels3, collapse = " + "))
dummies <- dummyVars(formula.levels3, dt.class.ified.combine)
dt.class_ified_level3 <- predict(dummies, newdata = dt.class.ified.combine)
dt.class_ified_level3 <- data.table(dt.class_ified_level3)
dt.class_ified_level3 <- dt.class_ified_level3[, lapply(.SD, as.factor)]
sapply(dt.class_ified_level3, class) # all factors

# combine
dt.class.ified.combine <- data.table(dt.class.ified.combine[, !colnames.levels3, with = F], dt.class_ified_level3)
##########################
## 1.2 centre and scale ##
##########################
prep.class.ified.combine <- preProcess(dt.class.ified.combine[, !c("Id", "Response", "isTest"), with = F]
                                       # , method = c("range")
                                       , method = c("center", "scale")
                                       , verbose = T)
dt.class.ified.combine <- predict(prep.class.ified.combine, dt.class.ified.combine)

############################################################################################
## 2.0 save ################################################################################
############################################################################################
dt.preprocessed.combine <- dt.class.ified.combine
save(dt.preprocessed.combine, file = "data/data_preprocess/dt_proprocess_combine.RData")













