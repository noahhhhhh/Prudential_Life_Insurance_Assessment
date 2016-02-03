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
dt.class.ified.combine[, Ht_Group := as.factor(as.integer(cut2(dt.class.ified.combine$Ht, cuts = c(0, .2, .4, .6, 1))))]
colNominal <- c(colNominal, "Ht_Group")
# dt.class.ified.combine[, Ht := NULL]
# colNominal <- colNominal[!colNominal %in% "Ht"]

############################################################################################
## 3.0 Wt_Group ############################################################################
############################################################################################
# 5 groups
dt.class.ified.combine[, Wt_Group := as.factor(as.integer(cut2(dt.class.ified.combine$Wt, cuts = c(0, .2, .4, .6, 1))))]
colNominal <- c(colNominal, "Wt_Group")
# dt.class.ified.combine[, Wt := NULL]
# colNominal <- colNominal[!colNominal %in% "Wt"]

############################################################################################
## 4.0 BMI_Group ###########################################################################
############################################################################################
# 5 groups
dt.class.ified.combine[, BMI_Group := as.factor(as.integer(cut2(dt.class.ified.combine$BMI, cuts = c(0, .2, .4, .6, 1))))]
colNominal <- c(colNominal, "BMI_Group")
# dt.class.ified.combine[, BMI := NULL]
# colNominal <- colNominal[!colNominal %in% "BMI"]

############################################################################################
## 5.0 square, cube ########################################################################
############################################################################################
dt.class.ified.combine[, Ins_Age_2 := Ins_Age ^ 2]
dt.class.ified.combine[, Ht_2 := Ht ^ 2]
dt.class.ified.combine[, Wt_2 := Wt ^ 2]
dt.class.ified.combine[, BNI_2 := BMI ^ 2]
colContinuous <- c(colContinuous, "Ins_Age_2", "Ht_2", "Wt_2", "BNI_2")

dt.class.ified.combine[, Ins_Age_3 := Ins_Age ^ 3]
dt.class.ified.combine[, Ht_3 := Ht ^ 3]
dt.class.ified.combine[, Wt_3 := Wt ^ 3]
dt.class.ified.combine[, BNI_3 := BMI ^ 3]
colContinuous <- c(colContinuous, "Ins_Age_3", "Ht_3", "Wt_3", "BNI_3")

############################################################################################
## 6.0 t-sne ###############################################################################
############################################################################################
require(Rtsne)
mx.class.ified.combine <- model.matrix(Response ~., mx.class.ified.combine <- dt.class.ified.combine[, !c("Id", "isTest"), with = F])[, -1]
tsne.out <- Rtsne(mx.class.ified.combine
                  , check_duplicates = F
                  , pca = F
                  , verbose = T
                  , perplexity = 30
                  , theta = .5
                  , dims = 2)

plot(tsne.out$Y, col = dt.class.ified.combine[dt.class.ified.combine$Response != 0]$Response)

mx.tsne.out <- tsne.out$Y
save(mx.tsne.out, file = "data/data_meta/dt_tsne.RData")
load("data/data_meta/dt_tsne.RData")
dt.class.ified.combine[, tsne_1 := mx.tsne.out[, 1]]
dt.class.ified.combine[, tsne_2 := mx.tsne.out[, 2]]
colContinuous <- c(colContinuous, "tsne_1", "tsne_2")

############################################################################################
## 7.0 classDist ###########################################################################
############################################################################################
mx.class.ified.combine <- data.matrix(dt.class.ified.combine[, !c("Id", "isTest", "Response"), with = F])[, -1]
centroids <- classDist(mx.class.ified.combine, as.factor(dt.class.ified.combine$Response))

############################################################################################
## 8.0 BMI and Age #########################################################################
############################################################################################
dt.class.ified.combine[, Age_BMI := Ins_Age * BMI]
colContinuous <- c(colContinuous, "Age_BMI")

############################################################################################
## 9.0 Med_Keywords_Count ##################################################################
############################################################################################
colname.medKeywords <- names(dt.class.ified.combine)[grepl("Medical_Keyword_", names(dt.class.ified.combine))]
dt.medKeywords <- dt.class.ified.combine[, colname.medKeywords, with = F][, lapply(.SD, as.integer)]
dt.class.ified.combine[, Med_Keywords_Count := rowSums(dt.medKeywords) - 48]
colDiscrete <- c(colDiscrete, "Med_Keywords_Count")

############################################################################################
## 9.0 save ################################################################################
############################################################################################
dt.featureEngineed.combine <- dt.class.ified.combine
save(dt.featureEngineed.combine, colNominal, colDiscrete, colContinuous, file = "data/data_enginee/dt_featureEngineed_combine.RData")
