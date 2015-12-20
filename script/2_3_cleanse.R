rm(list = ls()); gc();
setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Prudential_Life_Insurance_Assessment/")
load("data/data_clean/dt_class_ified_combine.RData")
require(data.table)
require(caret)
############################################################################################
## 1.0 cleanse #############################################################################
############################################################################################
#####################################
## 1.1 zero and near zero variance ##
#####################################
nzv <- nearZeroVar(dt.class.ified.combine, saveMetrics = T, foreach = T)
nzv
unique(dt.class.ified.combine$Product_Info_7)