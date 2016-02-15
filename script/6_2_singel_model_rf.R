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
ind.train <- createDataPartition(dt.preprocessed.combine[isTest == 0]$Response, p = .8, list = F) # remember to change it to .66
dt.train <- dt.preprocessed.combine[isTest == 0][ind.train]
dt.valid <- dt.preprocessed.combine[isTest == 0][-ind.train]
dt.test <- dt.preprocessed.combine[isTest == 1]
dim(dt.train); dim(dt.valid); dim(dt.test)

x.train <- model.matrix(Response ~., dt.train[, !c("Id", "isTest"), with = F])[, -1]
y.train <- dt.train$Response
dmx.train <- xgb.DMatrix(data =  x.train, label = y.train)

x.valid <- model.matrix(Response ~., dt.valid[, !c("Id", "isTest"), with = F])[, -1]
y.valid <- dt.valid$Response
dmx.valid <- xgb.DMatrix(data =  x.valid, label = y.valid)

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

pred.valid <- predict(md.rf, dt.valid)
pred.valid <- predictions(pred.valid)

pred.test <- predict(md.rf, dt.test)
pred.test <- predictions(pred.test)

## save ##
save(pred.train, pred.valid, pred.test, ind.train, file = "model/rf.RData")

cat("optimising the cuts on pred.train ...\n")
set.seed(1024)
trainForOpt <- sample(length(pred.train), length(pred.train) * .8)
pred.train.forOpt <- pred.train[trainForOpt]
SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
    cuts <- c(min(pred.train.forOpt), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.train.forOpt))
    pred <- as.integer(cut2(pred.train.forOpt, cuts))
    err <- ScoreQuadraticWeightedKappa(pred, y.train[trainForOpt], 1, 8)
    return(-err)
}

optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
optCuts

cat("applying optCuts on valid ...\n")
cuts.valid <- c(min(pred.valid), optCuts$par, max(pred.valid))
pred.valid.op <- as.integer(cut2(pred.valid, cuts.valid))
score <- ScoreQuadraticWeightedKappa(y.valid, pred.valid.op)
score
# 0.6359023

cat("applying optCuts on test ...\n")
cuts.test <- c(min(pred.test), optCuts$par, max(pred.test))
pred.test.op <- as.integer(cut2(pred.test, cuts.test))










