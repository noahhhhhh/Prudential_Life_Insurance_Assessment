rm(list = ls()); gc();
setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Prudential_Life_Insurance_Assessment/")
load("data/data_enginee/dt_featureEngineed_combine.RData")
require(data.table)
require(caret)
############################################################################################
## 1.0 preprocess ##########################################################################
############################################################################################
# ##########################
# ## 1.1 nzv and nearZero ##
# ##########################
# nzv.train <- nearZeroVar(dt.featureEngineed.combine[isTest == 0, !c("Id", "Response", "isTest"), with = F], saveMetrics = T)
# nzv.test <- nearZeroVar(dt.featureEngineed.combine[isTest == 1,!c("Id", "Response", "isTest"), with = F], saveMetrics = T)
# 
# col.nzv.train <- rownames(nzv.train[nzv.train$nzv, ])
# length(col.nzv.train)
# # [1] 73
# # col.nzv.test <- rownames(nzv.test[nzv.test$nzv, ])
# # length(col.nzv.test)
# # [1] 70
# 
# # col.nzv <- union(col.nzv.test, col.nzv.train)
# # length(col.nzv)
# # 143
# 
# # exclude them (version 1)
# dt.featureEngineed.combine <- dt.featureEngineed.combine[, - col.nzv.train, with = F]
# dim(dt.featureEngineed.combine)
# # [1] 79146   114
# 
# # select them (version 2)
# # NX: to be continued

##########################
## 1.2 centre and scale ##
##########################
prep.class.ified.combine <- preProcess(dt.featureEngineed.combine[, !c("Id", "Response", "isTest"), with = F]
                                       , method = c("range")
                                       # , method = c("center", "scale")
                                       , verbose = T)
dt.featureEngineed.combine <- predict(prep.class.ified.combine, dt.featureEngineed.combine)

####################
## 1.3 dummyVsars ##
####################
# levels == 3
no.of.levels <- sapply(dt.featureEngineed.combine[, colNominal, with = F], function (x) {length(names(table(x)))})
colnames.levels3 <- names(no.of.levels)[no.of.levels >= 3]
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
dummies <- dummyVars(formula.levels3, dt.featureEngineed.combine)
dt.class_ified_level3 <- predict(dummies, newdata = dt.featureEngineed.combine)
dt.class_ified_level3 <- data.table(dt.class_ified_level3)
dt.class_ified_level3 <- dt.class_ified_level3[, lapply(.SD, as.factor)]
sapply(dt.class_ified_level3, class) # all factors
colnames.class_ified_level3 <- names(dt.class_ified_level3)
colnames.class_ified_level3
# [1] "Product_Info_7.1"      "Product_Info_7.2"      "Product_Info_7.3"      "InsuredInfo_1.1"      
# [5] "InsuredInfo_1.2"       "InsuredInfo_1.3"       "Insurance_History_2.1" "Insurance_History_2.2"
# [9] "Insurance_History_2.3" "Insurance_History_3.1" "Insurance_History_3.2" "Insurance_History_3.3"
# [13] "Insurance_History_4.1" "Insurance_History_4.2" "Insurance_History_4.3" "Insurance_History_7.1"
# [17] "Insurance_History_7.2" "Insurance_History_7.3" "Insurance_History_8.1" "Insurance_History_8.2"
# [21] "Insurance_History_8.3" "Insurance_History_9.1" "Insurance_History_9.2" "Insurance_History_9.3"
# [25] "Family_Hist_1.1"       "Family_Hist_1.2"       "Family_Hist_1.3"       "Medical_History_3.1"  
# [29] "Medical_History_3.2"   "Medical_History_3.3"   "Medical_History_5.1"   "Medical_History_5.2"  
# [33] "Medical_History_5.3"   "Medical_History_6.1"   "Medical_History_6.2"   "Medical_History_6.3"  
# [37] "Medical_History_7.1"   "Medical_History_7.2"   "Medical_History_7.3"   "Medical_History_8.1"  
# [41] "Medical_History_8.2"   "Medical_History_8.3"   "Medical_History_9.1"   "Medical_History_9.2"  
# [45] "Medical_History_9.3"   "Medical_History_11.1"  "Medical_History_11.2"  "Medical_History_11.3" 
# [49] "Medical_History_12.1"  "Medical_History_12.2"  "Medical_History_12.3"  "Medical_History_13.1" 
# [53] "Medical_History_13.2"  "Medical_History_13.3"  "Medical_History_14.1"  "Medical_History_14.2" 
# [57] "Medical_History_14.3"  "Medical_History_16.1"  "Medical_History_16.2"  "Medical_History_16.3" 
# [61] "Medical_History_17.1"  "Medical_History_17.2"  "Medical_History_17.3"  "Medical_History_18.1" 
# [65] "Medical_History_18.2"  "Medical_History_18.3"  "Medical_History_19.1"  "Medical_History_19.2" 
# [69] "Medical_History_19.3"  "Medical_History_20.1"  "Medical_History_20.2"  "Medical_History_20.3" 
# [73] "Medical_History_21.1"  "Medical_History_21.2"  "Medical_History_21.3"  "Medical_History_23.1" 
# [77] "Medical_History_23.2"  "Medical_History_23.3"  "Medical_History_25.1"  "Medical_History_25.2" 
# [81] "Medical_History_25.3"  "Medical_History_26.1"  "Medical_History_26.2"  "Medical_History_26.3" 
# [85] "Medical_History_27.1"  "Medical_History_27.2"  "Medical_History_27.3"  "Medical_History_28.1" 
# [89] "Medical_History_28.2"  "Medical_History_28.3"  "Medical_History_29.1"  "Medical_History_29.2" 
# [93] "Medical_History_29.3"  "Medical_History_3.01"  "Medical_History_3.02"  "Medical_History_3.03" 
# [97] "Medical_History_3.11"  "Medical_History_3.12"  "Medical_History_3.13"  "Medical_History_3.31" 
# [101] "Medical_History_3.32"  "Medical_History_3.33"  "Medical_History_3.41"  "Medical_History_3.42" 
# [105] "Medical_History_3.43"  "Medical_History_3.51"  "Medical_History_3.52"  "Medical_History_3.53" 
# [109] "Medical_History_3.61"  "Medical_History_3.62"  "Medical_History_3.63"  "Medical_History_3.71" 
# [113] "Medical_History_3.72"  "Medical_History_3.73"  "Medical_History_3.81"  "Medical_History_3.82" 
# [117] "Medical_History_3.83"  "Medical_History_3.91"  "Medical_History_3.92"  "Medical_History_3.93" 
# [121] "Medical_History_40.1"  "Medical_History_40.2"  "Medical_History_40.3"  "Medical_History_41.1" 
# [125] "Medical_History_41.2"  "Medical_History_41.3" 

colnames.class_ified_level3[94:120]
# [1] "Medical_History_3.01" "Medical_History_3.02" "Medical_History_3.03" "Medical_History_3.11"
# [5] "Medical_History_3.12" "Medical_History_3.13" "Medical_History_3.31" "Medical_History_3.32"
# [9] "Medical_History_3.33" "Medical_History_3.41" "Medical_History_3.42" "Medical_History_3.43"
# [13] "Medical_History_3.51" "Medical_History_3.52" "Medical_History_3.53" "Medical_History_3.61"
# [17] "Medical_History_3.62" "Medical_History_3.63" "Medical_History_3.71" "Medical_History_3.72"
# [21] "Medical_History_3.73" "Medical_History_3.81" "Medical_History_3.82" "Medical_History_3.83"
# [25] "Medical_History_3.91" "Medical_History_3.92" "Medical_History_3.93"

colnames.class_ified_level3[785:835] #
old <- gsub("\\.", "", colnames.class_ified_level3[785:835]) #
n <- 19
colnames.class_ified_level3[785:835] <- paste(substr(old, 1, n-1), ".", substr(old, n, nchar(old)), sep = "") #
colnames.class_ified_level3
# [1] "Product_Info_7.1"      "Product_Info_7.2"      "Product_Info_7.3"      "InsuredInfo_1.1"      
# [5] "InsuredInfo_1.2"       "InsuredInfo_1.3"       "Insurance_History_2.1" "Insurance_History_2.2"
# [9] "Insurance_History_2.3" "Insurance_History_3.1" "Insurance_History_3.2" "Insurance_History_3.3"
# [13] "Insurance_History_4.1" "Insurance_History_4.2" "Insurance_History_4.3" "Insurance_History_7.1"
# [17] "Insurance_History_7.2" "Insurance_History_7.3" "Insurance_History_8.1" "Insurance_History_8.2"
# [21] "Insurance_History_8.3" "Insurance_History_9.1" "Insurance_History_9.2" "Insurance_History_9.3"
# [25] "Family_Hist_1.1"       "Family_Hist_1.2"       "Family_Hist_1.3"       "Medical_History_3.1"  
# [29] "Medical_History_3.2"   "Medical_History_3.3"   "Medical_History_5.1"   "Medical_History_5.2"  
# [33] "Medical_History_5.3"   "Medical_History_6.1"   "Medical_History_6.2"   "Medical_History_6.3"  
# [37] "Medical_History_7.1"   "Medical_History_7.2"   "Medical_History_7.3"   "Medical_History_8.1"  
# [41] "Medical_History_8.2"   "Medical_History_8.3"   "Medical_History_9.1"   "Medical_History_9.2"  
# [45] "Medical_History_9.3"   "Medical_History_11.1"  "Medical_History_11.2"  "Medical_History_11.3" 
# [49] "Medical_History_12.1"  "Medical_History_12.2"  "Medical_History_12.3"  "Medical_History_13.1" 
# [53] "Medical_History_13.2"  "Medical_History_13.3"  "Medical_History_14.1"  "Medical_History_14.2" 
# [57] "Medical_History_14.3"  "Medical_History_16.1"  "Medical_History_16.2"  "Medical_History_16.3" 
# [61] "Medical_History_17.1"  "Medical_History_17.2"  "Medical_History_17.3"  "Medical_History_18.1" 
# [65] "Medical_History_18.2"  "Medical_History_18.3"  "Medical_History_19.1"  "Medical_History_19.2" 
# [69] "Medical_History_19.3"  "Medical_History_20.1"  "Medical_History_20.2"  "Medical_History_20.3" 
# [73] "Medical_History_21.1"  "Medical_History_21.2"  "Medical_History_21.3"  "Medical_History_23.1" 
# [77] "Medical_History_23.2"  "Medical_History_23.3"  "Medical_History_25.1"  "Medical_History_25.2" 
# [81] "Medical_History_25.3"  "Medical_History_26.1"  "Medical_History_26.2"  "Medical_History_26.3" 
# [85] "Medical_History_27.1"  "Medical_History_27.2"  "Medical_History_27.3"  "Medical_History_28.1" 
# [89] "Medical_History_28.2"  "Medical_History_28.3"  "Medical_History_29.1"  "Medical_History_29.2" 
# [93] "Medical_History_29.3"  "Medical_History_30.1"  "Medical_History_30.2"  "Medical_History_30.3" 
# [97] "Medical_History_31.1"  "Medical_History_31.2"  "Medical_History_31.3"  "Medical_History_33.1" 
# [101] "Medical_History_33.2"  "Medical_History_33.3"  "Medical_History_34.1"  "Medical_History_34.2" 
# [105] "Medical_History_34.3"  "Medical_History_35.1"  "Medical_History_35.2"  "Medical_History_35.3" 
# [109] "Medical_History_36.1"  "Medical_History_36.2"  "Medical_History_36.3"  "Medical_History_37.1" 
# [113] "Medical_History_37.2"  "Medical_History_37.3"  "Medical_History_38.1"  "Medical_History_38.2" 
# [117] "Medical_History_38.3"  "Medical_History_39.1"  "Medical_History_39.2"  "Medical_History_39.3" 
# [121] "Medical_History_40.1"  "Medical_History_40.2"  "Medical_History_40.3"  "Medical_History_41.1" 
# [125] "Medical_History_41.2"  "Medical_History_41.3" 
colnames(dt.class_ified_level3) <- colnames.class_ified_level3
# combine
dt.featureEngineed.combine <- data.table(dt.featureEngineed.combine[, !colnames.levels3, with = F], dt.class_ified_level3)
colNominal <- c(colNominal, colnames.class_ified_level3)
colNominal <- colNominal[!colNominal %in% colnames.levels3]

##############################
## 1.4 kmeans meta features ##
##############################
colnames <- names(dt.featureEngineed.combine)
#####################
## Employment_Info ##
#####################
str(dt.featureEngineed.combine[, colnames[grep("Employment_Info", colnames)], with = F])
md.kmeans.employment_info <- kmeans(dt.featureEngineed.combine[, colnames[grep("Employment_Info", colnames)], with = F]
                                    , centers = 5
                                    , nstart = 20)
Employment_Info_Kmeans <- as.factor(md.kmeans.employment_info$cluster)

##################
## Product_Info ##
##################
str(dt.featureEngineed.combine[, colnames[grep("Product_Info", colnames)], with = F])
md.kmeans.product_info <- kmeans(dt.featureEngineed.combine[, colnames[grep("Product_Info", colnames)], with = F]
                                 , centers = 5
                                 , nstart = 20)
Product_Info_Kmeans <- as.factor(md.kmeans.product_info$cluster)

#################
## InsuredInfo ##
#################
str(dt.featureEngineed.combine[, colnames[grep("InsuredInfo", colnames)], with = F])
md.kmeans.insuredinfo <- kmeans(dt.featureEngineed.combine[, colnames[grep("InsuredInfo", colnames)], with = F]
                                , centers = 5
                                , nstart = 20)
InsuredInfo_Kmeans <- as.factor(md.kmeans.insuredinfo$cluster)

#######################
## Insurance_History ##
#######################
str(dt.featureEngineed.combine[, colnames[grep("Insurance_History", colnames)], with = F])
md.kmeans.insured_history <- kmeans(dt.featureEngineed.combine[, colnames[grep("Insurance_History", colnames)], with = F]
                                    , centers = 5
                                    , nstart = 20)
Insurance_History_Kmeans <- as.factor(md.kmeans.insured_history$cluster)

#################
## Family_Hist ##
#################
str(dt.featureEngineed.combine[, colnames[grep("Family_Hist", colnames)], with = F])
md.kmeans.family_hist <- kmeans(dt.featureEngineed.combine[, colnames[grep("Family_Hist", colnames)], with = F]
                                , centers = 5
                                , nstart = 20)
Family_Hist_Kmeans <- as.factor(md.kmeans.family_hist$cluster)

#####################
## Medical_History ##
#####################
str(dt.featureEngineed.combine[, colnames[grep("Medical_History", colnames)], with = F])
md.kmeans.medical_history <- kmeans(dt.featureEngineed.combine[, colnames[grep("Medical_History", colnames)], with = F]
                                    , centers = 5
                                    , nstart = 20)
Medical_History_Kmeans <- as.factor(md.kmeans.medical_history$cluster)

#####################
## Medical_Keyword ##
#####################
str(dt.featureEngineed.combine[, colnames[grep("Medical_Keyword", colnames)], with = F])
md.kmeans.medical_keyword <- kmeans(dt.featureEngineed.combine[, colnames[grep("Medical_Keyword", colnames)], with = F]
                                    , centers = 5
                                    , nstart = 20)
Medical_Keyword_Kmeans <- as.factor(md.kmeans.medical_keyword$cluster)

#########################################
## 1.4 add the kmeans meta features in ##
#########################################
load("data/data_meta/kmeans.RData")
dt.featureEngineed.combine[, Employment_Info_Kmeans := Employment_Info_Kmeans]
dt.featureEngineed.combine[, Product_Info_Kmeans := Product_Info_Kmeans]
dt.featureEngineed.combine[, InsuredInfo_Kmeans := InsuredInfo_Kmeans]
dt.featureEngineed.combine[, Insurance_History_Kmeans := Insurance_History_Kmeans]
dt.featureEngineed.combine[, Family_Hist_Kmeans := Family_Hist_Kmeans]
dt.featureEngineed.combine[, Medical_History_Kmeans := Medical_History_Kmeans]
dt.featureEngineed.combine[, Medical_Keyword_Kmeans := Medical_Keyword_Kmeans]

#############
## 1.5 pca ##
#############
#########
## all ##
#########
dt.pca.temp <- dt.featureEngineed.combine[, !c("isTest", "Response", "Id"), with = F]
dt.pca.temp <- dt.pca.temp[, lapply(.SD, as.numeric)]
md.pca <- prcomp(dt.pca.temp, scale. = F)


###############
## 1.6 noise ##
###############



############################################################################################
## 2.0 save ################################################################################
############################################################################################
save(Employment_Info_Kmeans
     , Product_Info_Kmeans
     , InsuredInfo_Kmeans
     , Insurance_History_Kmeans
     , Family_Hist_Kmeans
     , Medical_History_Kmeans
     , Medical_Keyword_Kmeans
     , file = "data/data_meta/kmeans.RData")

dt.preprocessed.combine <- dt.featureEngineed.combine
save(dt.preprocessed.combine, file = "data/data_preprocess/dt_proprocess_combine.RData")













