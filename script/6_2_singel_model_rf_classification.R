rm(list = ls()); gc();
setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Prudential_Life_Insurance_Assessment/")
load("data/data_preprocess/dt_proprocess_combine.RData")
source("script/utilities/metrics.R")
require(data.table)
require(caret)
require(Metrics)
require(Hmisc)
############################################################################################
## 1.0 random forest #######################################################################
############################################################################################
################################
## 1.1 train, valid, and test ##
################################
require(xgboost)
require(Ckmeans.1d.dp)
cat("prepare train, valid, and test data set...\n")
set.seed(888)
ind.train <- createDataPartition(dt.preprocessed.combine[isTest == 0]$Response, p = .9, list = F) # remember to change it to .66
dt.train <- dt.preprocessed.combine[isTest == 0][ind.train]
dt.valid <- dt.preprocessed.combine[isTest == 0][-ind.train]
set.seed(888)
ind.valid <- createDataPartition(dt.valid$Response, p = .5, list = F)
dt.valid1 <- dt.valid[ind.valid]
dt.valid2 <- dt.valid[-ind.valid]
dt.test <- dt.preprocessed.combine[isTest == 1]
dim(dt.train); dim(dt.valid); dim(dt.test)

# apply noise on dt.train
# dim(dt.train)
# dt.train <- Noise(dt.train, noise_l = 0, noise_u = .00005, col_excl = c(colNominal, "Id", "Response", "isTest"))
# dim(dt.train)
x.train <- model.matrix(Response ~., dt.train[, !c("Id", "isTest"), with = F])[, -1]
y.train <- dt.train$Response
dmx.train <- xgb.DMatrix(data =  x.train, label = y.train)

x.valid <- model.matrix(Response ~., dt.valid[, !c("Id", "isTest"), with = F])[, -1]
y.valid <- dt.valid$Response
dmx.valid <- xgb.DMatrix(data =  x.valid, label = y.valid)

x.valid1 <- model.matrix(Response ~., dt.valid1[, !c("Id", "isTest"), with = F])[, -1]
y.valid1 <- dt.valid1$Response
dmx.valid1 <- xgb.DMatrix(data =  x.valid1, label = y.valid1)

x.valid2 <- model.matrix(Response ~., dt.valid2[, !c("Id", "isTest"), with = F])[, -1]
y.valid2 <- dt.valid2$Response
dmx.valid2 <- xgb.DMatrix(data =  x.valid2, label = y.valid2)

x.test <- model.matrix(~., dt.preprocessed.combine[isTest == 1, !c("Id", "isTest", "Response"), with = F])[, -1]

################################
## 1.2 train ###################
################################
require(ranger)
ls.pred.train.encode <- list()
ls.pred.valid.encode <- list()
ls.pred.test.encode <- list()
for(r in 1:8){
    y.train.encode <- ifelse(dt.train$Response == r, 1, 0)
    dt.train.rf <- dt.train[, !c("Id", "isTest", "Response"), with = F]
    dt.train.rf <- dt.train.rf[, Response := y.train.encode]
    
    set.seed(888)
    cat("train rf ...\n")
    md.rf <- ranger(Response ~.
                    , data = dt.train.rf
                    , num.trees = 3500
                    , mtry = 12
                    , importance = "impurity"
                    , write.forest = T
                    # , probability = T
                    , min.node.size = 20
                    , num.threads = 8
                    , verbose = T
    )
    cat("\npredicting on train, valid, and test encode ...\n")
    pred.train.encode <- predictions(predict(md.rf, dt.train.rf))
    pred.valid.encode <- predictions(predict(md.rf, dt.valid[, !c("Id", "isTest"), with = F]))
    pred.test.encode <- predictions(predict(md.rf, dt.test[, !c("Id", "isTest"), with = F]))
    
    ls.pred.train.encode[[r]] <- pred.train.encode
    ls.pred.valid.encode[[r]] <- pred.valid.encode
    ls.pred.test.encode[[r]] <- pred.test.encode
    
}

cat("get the probability of each response")
dt.pred.train <- as.data.table(sapply(ls.pred.train.encode, print))
pred.train <- apply(dt.pred.train, 1, which.max)
ScoreQuadraticWeightedKappa(pred.train, y.train)
# [1] 0.9144764
dt.valid.train <- as.data.table(sapply(ls.pred.valid.encode, print))
pred.valid <- apply(dt.valid.train, 1, which.max)
ScoreQuadraticWeightedKappa(pred.valid, y.valid)
# [1] 0.5414258

p8.train <- 1 - dt.pred.train$V7
p7.train <- dt.pred.train$V7 - dt.pred.train$V6
p6.train <- dt.pred.train$V6 - dt.pred.train$V5
p5.train <- dt.pred.train$V5 - dt.pred.train$V4
p4.train <- dt.pred.train$V4 - dt.pred.train$V3
p3.train <- dt.pred.train$V3 - dt.pred.train$V2
p2.train <- dt.pred.train$V2 - dt.pred.train$V1
p1.train <- dt.pred.train$V1

dt.train.result <- data.table(p1.train
                              , p2.train
                              , p3.train
                              , p4.train
                              , p5.train
                              , p6.train
                              , p7.train
                              , p8.train)
pred.train <- apply(dt.train.result, 1, which.max)
ScoreQuadraticWeightedKappa(pred.train, y.train)
# 0.9128559

dt.pred.valid <- as.data.table(sapply(ls.pred.valid.encode, print))
p8.valid <- 1 - dt.pred.valid$V7
p7.valid <- dt.pred.valid$V7 - dt.pred.valid$V6
p6.valid <- dt.pred.valid$V6 - dt.pred.valid$V5
p5.valid <- dt.pred.valid$V5 - dt.pred.valid$V4
p4.valid <- dt.pred.valid$V4 - dt.pred.valid$V3
p3.valid <- dt.pred.valid$V3 - dt.pred.valid$V2
p2.valid <- dt.pred.valid$V2 - dt.pred.valid$V1
p1.valid <- dt.pred.valid$V1

dt.valid.result <- data.table(p1.valid
                              , p2.valid
                              , p3.valid
                              , p4.valid
                              , p5.valid
                              , p6.valid
                              , p7.valid
                              , p8.valid)
pred.valid <- apply(dt.valid.result, 1, which.max)
ScoreQuadraticWeightedKappa(pred.valid, y.valid)
# 0.5465477

pred.train <- apply(dt.pred.train, 1, which.max)
ScoreQuadraticWeightedKappa(pred.train, y.train)
