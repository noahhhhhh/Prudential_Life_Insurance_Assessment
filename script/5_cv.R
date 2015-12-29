rm(list = ls()); gc();
setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Prudential_Life_Insurance_Assessment/")
load("data/data_preprocess/dt_proprocess_combine.RData")
source("script/utilities/metrics.R")
require(data.table)
require(caret)
require(Metrics)
require(Hmisc)
############################################################################################
## 1.0 xgboost - gbtree ####################################################################
############################################################################################
################################
## 1.1 train, valid, and test ##
################################
require(xgboost)
require(Ckmeans.1d.dp)
cat("prepare train, valid, and test data set\n")
set.seed(888)
ind.train <- createDataPartition(dt.preprocessed.combine[isTest == 0]$Response, p = .9, list = F)
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
## 1.2 xgb with n fold #########
################################
cat("cross validate xgb\n")
set.seed(888)
# create a 3 folds
folds <- createFolds(dt.train$Response, k = 3, list = F)

# set up the parameters
# 1: m
cv.booster <- c("gbtree", "gblinear")
# 2: n
cv.eta <- c(.001, .025, .05, .075, .1, .2, .5)
cv.nroudns <- c(8000, 5000, 3000, 2000, 1500, 1200, 1000)
cv.print.every.n <- c(250, 150, 100, 75, 50, 40, 30)
# 3: n
cv.min_child_weight <- c(10, 20, 20, 30, 30, 40, 40)
cv.max_depth <- rep(8, 7)
cv.subsample <- c(.9, .8, .8, .7, .7, .6, .6)
cv.colsample_bytree <- c(.8, .8, .7, .7, .6, .6, .5)

# set up vector n
vec.n <- as.numeric()
# set up vecor m
vec.m <- as.numeric()
# set up vector score
vec.score <- as.numeric()

# setup the predictions
# train a model on folds

for(m in 1:2){ # boosters - param 1
    for(n in 1:7){ # parameters - param 2 and 3
        # set up a score metric for folds
        score.folds <- 0
        for(k in 1:3){ # folds
            set.seed(m * 8 + n * 64 + k * 512)
            # dmx.train.fold
            dt.train.fold <- dt.train[folds != k]
            x.train.fold <- model.matrix(Response ~., dt.train.fold[, !c("Id", "isTest"), with = F])[, -1]
            y.train.fold <- dt.train.fold$Response
            dmx.train.fold <- xgb.DMatrix(data =  x.train.fold, label = y.train.fold)
            # dmx.valid.fold
            dt.valid.fold <- dt.train[folds == k]
            x.valid.fold <- model.matrix(Response ~., dt.valid.fold[, !c("Id", "isTest"), with = F])[, -1]
            y.valid.fold <- dt.valid.fold$Response
            dmx.valid.fold <- xgb.DMatrix(data =  x.valid.fold, label = y.valid.fold)
            # train
            cv.xgb.out <- xgb.train(data = dmx.train.fold
                                    , booster = cv.booster[m]
                                    , objective = "reg:linear"
                                    , params = list(nthread = 8
                                                    , eta = cv.eta[n]
                                                    , min_child_weight = cv.min_child_weight[n]
                                                    , max_depth = cv.max_depth[n]
                                                    , subsample = cv.subsample[n]
                                                    , colsample_bytree = cv.colsample_bytree[n]
                                                    , metrics = "rmse"
                                    )
                                    # , feval = QuadraticWeightedKappa
                                    , early.stop.round = 20
                                    , maximize = F
                                    , print.every.n = cv.print.every.n[n]
                                    , nrounds = cv.nroudns[n]
                                    , watchlist = list(valid = dmx.valid.fold, train = dmx.train.fold)
                                    , verbose = T
            )
            pred.valid.fold <- predict(cv.xgb.out, dmx.valid.fold)
            SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
                cuts <- c(min(pred.valid.fold), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.valid.fold))
                pred <- as.numeric(cut2(pred.valid.fold, cuts))
                err <- ScoreQuadraticWeightedKappa(pred, y.valid.fold, 1, 8)
                return(-err)
            }
            optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
            optCuts
            
            cuts <- c(min(pred.valid.fold), optCuts$par, max(pred.valid.fold))
            score <- ScoreQuadraticWeightedKappa(y.valid.fold, as.integer(cut2(pred.valid.fold, cuts)))
            score.folds <- score.folds + score / 3
        }
        vec.n <- c(vec.n, n)
        vec.m <- c(vec.m, m)
        vec.score <- c(vec.score, score.folds)
        print(paste("-----------m = ", m, "; n = ", n, "; score =", score.folds))
    }
}
dt.result <- data.table(booster = vec.m, params = vec.n, score = vec.score)
#    booster params     score
# 1:       1      1 0.6520153
# 2:       1      2 0.6554311 *
# 3:       1      3 0.6524763
# 4:       1      4 0.6504982
# 5:       1      5 0.6488586
# 6:       1      6 0.6407940
# 7:       1      7 0.6121644
# 8:       2      1 0.6051335
# 9:       2      2 0.6118097
# 10:       2      3 0.6134282
# 11:       2      4 0.6112849
# 12:       2      5 0.6119529
# 13:       2      6 0.6102465
# 14:       2      7 0.6138835
# m == 1; n == 2 produced the bset result
# reproduce it
m <- 1; n <- 2
set.seed(m * 8 + n * 64 + k * 512)
md.xgb <- xgb.train(data = dmx.train
                    , booster = cv.booster[m]
                    , objective = "reg:linear"
                    , params = list(nthread = 8
                                    , eta = cv.eta[n]
                                    , min_child_weight = cv.min_child_weight[n]
                                    , max_depth = cv.max_depth[n]
                                    , subsample = cv.subsample[n]
                                    , colsample_bytree = cv.colsample_bytree[n]
                                    , metrics = "rmse"
                    )
                    # , feval = QuadraticWeightedKappa
                    , early.stop.round = 20
                    , maximize = F
                    , print.every.n = cv.print.every.n[n]
                    , nrounds = cv.nroudns[n]
                    , watchlist = list(valid = dmx.valid.fold, train = dmx.train.fold)
                    , verbose = T
)
pred.train <- predict(md.xgb, dmx.train)
SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
    cuts <- c(min(pred.train), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.train))
    pred <- as.integer(cut2(pred.train, cuts))
    err <- ScoreQuadraticWeightedKappa(pred, y.train, 1, 8)
    return(-err)
}
optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
optCuts

cuts <- c(min(pred.train), optCuts$par, max(pred.train))
score <- ScoreQuadraticWeightedKappa(y.train, as.integer(cut2(pred.train, cuts)))












