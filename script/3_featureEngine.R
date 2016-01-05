rm(list = ls()); gc();
setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Prudential_Life_Insurance_Assessment/")
load("data/data_clean/dt_class_ified_combine.RData")
require(data.table)
############################################################################################
## 1.0 Age_Group ###########################################################################
############################################################################################
# 5 groups
dt.class.ified.combine[, Age_Group := as.factor(as.integer(cut2(dt.class.ified.combine$Ins_Age, cuts = c(0, .2, .4, .6, 1))))]
colNominal <- c(colNominal, "Age_Group")
# dt.class.ified.combine[, Ins_Age := NULL]
# colNominal <- colNominal[!colNominal %in% "Ins_Age"]

############################################################################################
## 2.0 Ht_Group ############################################################################
############################################################################################
# 5 groups
dt.class.ified.combine[, Ht_Group := as.factor(as.integer(cut2(dt.class.ified.combine$Ht, cuts = c(.6, .8, 1))))]
colNominal <- c(colNominal, "Ht_Group")
# dt.class.ified.combine[, Ht := NULL]
# colNominal <- colNominal[!colNominal %in% "Ht"]

############################################################################################
## 3.0 Wt_Group ############################################################################
############################################################################################
# 5 groups
dt.class.ified.combine[, Wt_Group := as.factor(as.integer(cut2(dt.class.ified.combine$Wt, cuts = c(0, .2, .4))))]
colNominal <- c(colNominal, "Wt_Group")
# dt.class.ified.combine[, Wt := NULL]
# colNominal <- colNominal[!colNominal %in% "Wt"]

############################################################################################
## 4.0 BMI_Group ############################################################################
############################################################################################
# 5 groups
dt.class.ified.combine[, BMI_Group := as.factor(as.integer(cut2(dt.class.ified.combine$BMI, cuts = c(.4, .6))))]
colNominal <- c(colNominal, "BMI_Group")
# dt.class.ified.combine[, BMI := NULL]
# colNominal <- colNominal[!colNominal %in% "BMI"]

############################################################################################
## 6.0 save ################################################################################
############################################################################################
dt.featureEngineed.combine <- dt.class.ified.combine
save(dt.featureEngineed.combine, colNominal, colDiscrete, colContinuous, file = "data/data_enginee/dt_featureEngineed_combine.RData")
