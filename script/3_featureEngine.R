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
## 6.0 BMI and Age #########################################################################
############################################################################################
dt.class.ified.combine[, Age_BMI := Ins_Age * BMI]
colContinuous <- c(colContinuous, "Age_BMI")

############################################################################################
## 7.0 Med_Keywords_Count ##################################################################
############################################################################################
colname.medKeywords <- names(dt.class.ified.combine)[grepl("Medical_Keyword_", names(dt.class.ified.combine))]
dt.medKeywords <- dt.class.ified.combine[, colname.medKeywords, with = F][, lapply(.SD, as.integer)]
dt.class.ified.combine[, Med_Keywords_Count := rowSums(dt.medKeywords) - 48]
colDiscrete <- c(colDiscrete, "Med_Keywords_Count")

############################################################################################
## 8.0 kmeans ##############################################################################
############################################################################################
colnames <- names(dt.class.ified.combine)
## scale
prep.class.ified.combine <- preProcess(dt.class.ified.combine[, !c("Id", "Response", "isTest"), with = F]
                                       # , method = c("range")
                                       , method = c("center", "scale")
                                       , verbose = T)
dt.class.ified.combine.scale <- predict(prep.class.ified.combine, dt.class.ified.combine)

#####################
## Employment_Info ##
#####################
cat("kmeans of Employment_Info ...")
set.seed(888)
md.kmeans.employment_info <- kmeans(dt.class.ified.combine.scale[, colnames[grep("Employment_Info", colnames)], with = F]
                                    , centers = 8
                                    , nstart = 20)
Employment_Info_Kmeans <- md.kmeans.employment_info$cluster

##################
## Product_Info ##
##################
cat("kmeans of Product_Info ...")
set.seed(888)
md.kmeans.product_info <- kmeans(dt.class.ified.combine.scale[, colnames[grep("Product_Info", colnames)], with = F]
                                 , centers = 8
                                 , nstart = 20)
Product_Info_Kmeans <- md.kmeans.product_info$cluster

#################
## InsuredInfo ##
#################
cat("kmeans of InsuredInfo ...")
set.seed(888)
md.kmeans.insuredinfo <- kmeans(dt.class.ified.combine.scale[, colnames[grep("InsuredInfo", colnames)], with = F]
                                , centers = 8
                                , nstart = 20)
InsuredInfo_Kmeans <- md.kmeans.insuredinfo$cluster

#######################
## Insurance_History ##
#######################
cat("kmeans of Insurance_History ...")
set.seed(888)
md.kmeans.insured_history <- kmeans(dt.class.ified.combine.scale[, colnames[grep("Insurance_History", colnames)], with = F]
                                    , centers = 8
                                    , nstart = 20)
Insurance_History_Kmeans <- md.kmeans.insured_history$cluster

#################
## Family_Hist ##
#################
cat("kmeans of Family_Hist ...")
set.seed(888)
md.kmeans.family_hist <- kmeans(dt.class.ified.combine.scale[, colnames[grep("Family_Hist", colnames)], with = F]
                                , centers = 8
                                , nstart = 20)
Family_Hist_Kmeans <- md.kmeans.family_hist$cluster

#####################
## Medical_History ##
#####################
cat("kmeans of Medical_History ...")
set.seed(888)
md.kmeans.medical_history <- kmeans(dt.class.ified.combine.scale[, colnames[grep("Medical_History", colnames)], with = F]
                                    , centers = 8
                                    , nstart = 20)
Medical_History_Kmeans <- md.kmeans.medical_history$cluster

#####################
## Medical_Keyword ##
#####################
cat("kmeans of Medical_Keyword ...")
set.seed(888)
md.kmeans.medical_keyword <- kmeans(dt.class.ified.combine.scale[, colnames[grep("Medical_Keyword", colnames)], with = F]
                                    , centers = 8
                                    , nstart = 20)
Medical_Keyword_Kmeans <- md.kmeans.medical_keyword$cluster

#########
## All ##
#########
cat("kmeans of all ...")
set.seed(888)
md.kmeans.all <- kmeans(dt.class.ified.combine.scale[, !c("Id", "Response", "isTest"), with = F]
                                    , centers = 8
                                    , nstart = 20)
All_Kmeans <- md.kmeans.all$cluster

save(Employment_Info_Kmeans
     , Product_Info_Kmeans
     , InsuredInfo_Kmeans
     , Insurance_History_Kmeans
     , Family_Hist_Kmeans
     , Medical_History_Kmeans
     , Medical_Keyword_Kmeans
     , All_Kmeans
     , file = "data/data_meta/kmeans.RData")

#####################################
## add the kmeans meta features in ##
#####################################
load("data/data_meta/kmeans.RData")
dt.class.ified.combine[, Employment_Info_Kmeans := Employment_Info_Kmeans]
dt.class.ified.combine[, Product_Info_Kmeans := Product_Info_Kmeans]
dt.class.ified.combine[, InsuredInfo_Kmeans := InsuredInfo_Kmeans]
dt.class.ified.combine[, Insurance_History_Kmeans := Insurance_History_Kmeans]
dt.class.ified.combine[, Family_Hist_Kmeans := Family_Hist_Kmeans]
dt.class.ified.combine[, Medical_History_Kmeans := Medical_History_Kmeans]
dt.class.ified.combine[, Medical_Keyword_Kmeans := Medical_Keyword_Kmeans]
dt.class.ified.combine[, All_Kmeans := All_Kmeans]

colNominal <- c(colNominal, "Employment_Info_Kmeans", "Product_Info_Kmeans"
                , "InsuredInfo_Kmeans", "Insurance_History_Kmeans"
                , "Family_Hist_Kmeans", "Medical_History_Kmeans"
                , "Medical_Keyword_Kmeans", "All_Kmeans")

############################################################################################
## 9.0 t-sne ###############################################################################
############################################################################################
## scale
prep.class.ified.combine <- preProcess(dt.class.ified.combine[, !c("Id", "Response", "isTest"), with = F]
                                       # , method = c("range")
                                       , method = c("center", "scale")
                                       , verbose = T)
dt.class.ified.combine.scale <- predict(prep.class.ified.combine, dt.class.ified.combine)
## t-sne
require(Rtsne)
mx.class.ified.combine.scale <- data.matrix(dt.class.ified.combine.scale[, !c("Id", "isTest", "Response"), with = F])
tsne.out <- Rtsne(mx.class.ified.combine.scale
                  , check_duplicates = F
                  , pca = F
                  , verbose = T
                  , perplexity = 30
                  , theta = .5
                  , dims = 2)
plot(tsne.out$Y, col = dt.class.ified.combine[dt.class.ified.combine$Response != 0]$Response)
mx.tsne.out <- tsne.out$Y

#####################
## Employment_Info ##
#####################
cat("tsne of Employment_Info ...")
set.seed(888)
mx.class.ified.combine.scale.Employment_Info <- data.matrix(dt.class.ified.combine.scale[, colnames[grep("Employment_Info", colnames)], with = F])
tsne.out.Employment_Info <- Rtsne(mx.class.ified.combine.scale.Employment_Info
                  , check_duplicates = F
                  , pca = F
                  , verbose = T
                  , perplexity = 30
                  , theta = .5
                  , dims = 2)
plot(tsne.out.Employment_Info$Y, col = dt.class.ified.combine[dt.class.ified.combine$Response != 0]$Response)
mx.tsne.out.Employment_Info <- tsne.out.Employment_Info$Y

##################
## Product_Info ##
##################
cat("tsne of Product_Info ...")
set.seed(888)
mx.class.ified.combine.scale.Product_Info <- data.matrix(dt.class.ified.combine.scale[, colnames[grep("Product_Info", colnames)], with = F])
tsne.out.Product_Info <- Rtsne(mx.class.ified.combine.scale.Product_Info
                                  , check_duplicates = F
                                  , pca = F
                                  , verbose = T
                                  , perplexity = 30
                                  , theta = .5
                                  , dims = 2)
plot(tsne.out.Product_Info$Y, col = dt.class.ified.combine[dt.class.ified.combine$Response != 0]$Response)
mx.tsne.out.Product_Info <- tsne.out.Product_Info$Y

#################
## InsuredInfo ##
#################
cat("tsne of InsuredInfo ...")
set.seed(888)
mx.class.ified.combine.scale.InsuredInfo <- data.matrix(dt.class.ified.combine.scale[, colnames[grep("InsuredInfo", colnames)], with = F])
tsne.out.InsuredInfo <- Rtsne(mx.class.ified.combine.scale.InsuredInfo
                               , check_duplicates = F
                               , pca = F
                               , verbose = T
                               , perplexity = 30
                               , theta = .5
                               , dims = 2)
plot(tsne.out.InsuredInfo$Y, col = dt.class.ified.combine[dt.class.ified.combine$Response != 0]$Response)
mx.tsne.out.InsuredInfo <- tsne.out.InsuredInfo$Y

#######################
## Insurance_History ##
#######################
cat("tsne of Insurance_History ...")
set.seed(888)
mx.class.ified.combine.scale.Insurance_History <- data.matrix(dt.class.ified.combine.scale[, colnames[grep("Insurance_History", colnames)], with = F])
tsne.out.Insurance_History <- Rtsne(mx.class.ified.combine.scale.Insurance_History
                               , check_duplicates = F
                               , pca = F
                               , verbose = T
                               , perplexity = 30
                               , theta = .5
                               , dims = 2)
plot(tsne.out.Insurance_History$Y, col = dt.class.ified.combine[dt.class.ified.combine$Response != 0]$Response)
mx.tsne.out.Insurance_History <- tsne.out.Insurance_History$Y

#################
## Family_Hist ##
#################
cat("tsne of Family_Hist ...")
set.seed(888)
mx.class.ified.combine.scale.Family_Hist <- data.matrix(dt.class.ified.combine.scale[, colnames[grep("Family_Hist", colnames)], with = F])
tsne.out.Family_Hist <- Rtsne(mx.class.ified.combine.scale.Family_Hist
                               , check_duplicates = F
                               , pca = F
                               , verbose = T
                               , perplexity = 30
                               , theta = .5
                               , dims = 2)
plot(tsne.out.Family_Hist$Y, col = dt.class.ified.combine[dt.class.ified.combine$Response != 0]$Response)
mx.tsne.out.Family_Hist <- tsne.out.Family_Hist$Y

#####################
## Medical_History ##
#####################
cat("tsne of Medical_History ...")
set.seed(888)
mx.class.ified.combine.scale.Medical_History <- data.matrix(dt.class.ified.combine.scale[, colnames[grep("Medical_History", colnames)], with = F])
tsne.out.Medical_History <- Rtsne(mx.class.ified.combine.scale.Medical_History
                               , check_duplicates = F
                               , pca = F
                               , verbose = T
                               , perplexity = 30
                               , theta = .5
                               , dims = 2)
plot(tsne.out.Medical_History$Y, col = dt.class.ified.combine[dt.class.ified.combine$Response != 0]$Response)
mx.tsne.out.Medical_History <- tsne.out.Medical_History$Y

#####################
## Medical_Keyword ##
#####################
cat("tsne of Medical_Keyword ...")
set.seed(888)
mx.class.ified.combine.scale.Medical_Keyword <- data.matrix(dt.class.ified.combine.scale[, colnames[grep("Medical_Keyword", colnames)], with = F])
tsne.out.Medical_Keyword<- Rtsne(mx.class.ified.combine.scale.Medical_Keyword
                               , check_duplicates = F
                               , pca = F
                               , verbose = T
                               , perplexity = 30
                               , theta = .5
                               , dims = 2)
plot(tsne.out.Medical_Keyword$Y, col = dt.class.ified.combine[dt.class.ified.combine$Response != 0]$Response)
mx.tsne.out.Medical_Keyword <- tsne.out.Medical_Keyword$Y

save(mx.tsne.out
     , mx.tsne.out.Employment_Info
     , mx.tsne.out.Product_Info
     , mx.tsne.out.InsuredInfo
     , mx.tsne.out.Insurance_History
     , mx.tsne.out.Family_Hist
     , mx.tsne.out.Medical_History
     , mx.tsne.out.Medical_Keyword
     , file = "data/data_meta/dt_tsne.RData")
load("data/data_meta/dt_tsne.RData")
dt.class.ified.combine[, tsne_1 := mx.tsne.out[, 1]]
dt.class.ified.combine[, tsne_2 := mx.tsne.out[, 2]]
dt.class.ified.combine[, tsne_1_Employment_Info := mx.tsne.out.Employment_Info[, 1]]
dt.class.ified.combine[, tsne_2_Employment_Info := mx.tsne.out.Employment_Info[, 2]]
dt.class.ified.combine[, tsne_1_Product_Info := mx.tsne.out.Product_Info[, 1]]
dt.class.ified.combine[, tsne_2_Product_Info := mx.tsne.out.Product_Info[, 2]]
dt.class.ified.combine[, tsne_1_InsuredInfo := mx.tsne.out.InsuredInfo[, 1]]
dt.class.ified.combine[, tsne_2_InsuredInfo := mx.tsne.out.InsuredInfo[, 2]]
dt.class.ified.combine[, tsne_1_Insurance_History := mx.tsne.out.Insurance_History[, 1]]
dt.class.ified.combine[, tsne_2_Insurance_History := mx.tsne.out.Insurance_History[, 2]]
dt.class.ified.combine[, tsne_1_Family_Hist := mx.tsne.out.Family_Hist[, 1]]
dt.class.ified.combine[, tsne_2_Family_Hist := mx.tsne.out.Family_Hist[, 2]]
dt.class.ified.combine[, tsne_1_Medical_History := mx.tsne.out.Medical_History[, 1]]
dt.class.ified.combine[, tsne_2_Medical_History := mx.tsne.out.Medical_History[, 2]]
dt.class.ified.combine[, tsne_1_Medical_Keyword := mx.tsne.out.Medical_Keyword[, 1]]
dt.class.ified.combine[, tsne_2_Medical_Keyword := mx.tsne.out.Medical_Keyword[, 2]]
colContinuous <- c(colContinuous, "tsne_1", "tsne_2"
                   , "tsne_1_Employment_Info", "tsne_2_Employment_Info"
                   , "tsne_1_Product_Info", "tsne_2_Product_Info"
                   , "tsne_1_InsuredInfo", "tsne_2_InsuredInfo"
                   , "tsne_1_Insurance_History", "tsne_2_Insurance_History"
                   , "tsne_1_Family_Hist", "tsne_2_Family_Hist"
                   , "tsne_1_Medical_History", "tsne_2_Medical_History"
                   , "tsne_1_Medical_Keyword", "tsne_2_Medical_Keyword")

############################################################################################
## 9.0 save ################################################################################
############################################################################################
dt.featureEngineed.combine <- dt.class.ified.combine
save(dt.featureEngineed.combine, colNominal, colDiscrete, colContinuous, file = "data/data_enginee/dt_featureEngineed_combine.RData")
