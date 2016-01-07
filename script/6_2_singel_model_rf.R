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
dt.train.rf <- dt.train[, !c("Id", "isTest"), with = F]

set.seed(888)
cat("train rf ...\n")
md.rf <- ranger(Response ~.
                , data = dt.train.rf
                , num.trees = 3500
                , mtry = 36
                , importance = "impurity"
                , write.forest = T
                , min.node.size = 20
                , num.threads = 8
                , verbose = T
)

pred.train <- predict(md.rf, dt.train)
pred.train <- predictions(pred.train)
# pred.train <- as.integer(pred.train)

pred.valid <- predict(md.rf, dt.valid)
pred.valid <- predictions(pred.valid)
# pred.valid <- as.integer(pred.valid)

pred.test <- predict(md.rf, dt.test)
pred.test <- predictions(pred.test)
# pred.test <- as.integer(pred.test)

cat("optimising the cuts on pred.train ...\n")
SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
    cuts <- c(min(pred.train), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.train))
    pred <- as.integer(cut2(pred.train, cuts))
    err <- ScoreQuadraticWeightedKappa(pred, y.train, 1, 8)
    return(-err)
}

optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
optCuts

cat("applying optCuts on valid ...\n")
cuts.valid <- c(min(pred.valid), optCuts$par, max(pred.valid))
pred.valid.op <- as.integer(cut2(pred.valid, cuts.valid))
score <- ScoreQuadraticWeightedKappa(y.valid, pred.valid.op)
score
# [1] 0.6265719 (originally .550676; num.trees = 500, regression tree on raw features)
# [1] 0.5286518 (originally .5286518; num.trees = 500, classification tree on raw features)
# [1] 0.6444031 (originally .5558768; num.trees = 5000, regression tree on raw features)
# [1] 0.6450072 (originally .5546783; num.trees = 3500, mtry = 36, regression tree on 4 new group feature) *

cat("applying optCuts on test ...\n")
cuts.test <- c(min(pred.test), optCuts$par, max(pred.test))
pred.test.op <- as.integer(cut2(pred.test, cuts.test))











