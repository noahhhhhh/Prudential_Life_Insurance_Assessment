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
cv.nrounds <- c(12000, 8000, 6000, 5000, 4000, 3000, 2500)
cv.print.every.n <- c(250, 150, 100, 75, 50, 40, 30)
# 3: individuals
cv.min_child_weight <- c(10, 20, 30, 40, 50, 60, 70)
cv.max_depth <- rep(5, 6, 7, 8, 9, 10, 11)
cv.subsample <- c(.9, .8, .7, .6, .5, .4, .3)
cv.colsample_bytree <- c(.9, .8, .7, .6, .5, .4, .3)
cv.gamma <- c(0, .1, .2, .3, .4, .5, .6)

# set up vecor m
vec.m <- as.numeric()
# set up vector n
vec.n <- as.numeric()
# set up vector n.min_child_weight
vec.n.min_child_weight <- as.numeric()
# set up vector n.max_depth
vec.n.max_depth <- as.numeric()
# set up vector n.subsample
vec.n.subsample <- as.numeric()
# set up vector n.colsample_bytree
vec.n.colsample_bytree <- as.numeric()
# set up vector n.gamma
vec.n.gamma <- as.numeric()

# set up vector score
vec.score <- as.numeric()

# setup the predictions
# train a model on folds

for(m in 1:2){ # boosters
    for(n in 1:7){ # eta; nrounds; print.every.n
        for(n.min_child_weight in 1:7){ # min_child_weight
            for(n.max_depth in 1:7){ # max_depth
                for(n.subsample in 1:7){ # subsample
                    for(n.colsample_bytree in 1:7){ # colsample_bytree
                        for(n.gamma in 1:7){
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
                                set.seed(m * 8 + n * 64 + k * 512)
                                cv.xgb.out <- xgb.train(data = dmx.train.fold
                                                        , booster = cv.booster[m]
                                                        , objective = "count:poisson"
                                                        , params = list(nthread = 8
                                                                        , eta = cv.eta[n]
                                                                        , min_child_weight = cv.min_child_weight[n.min_child_weight]
                                                                        , max_depth = cv.max_depth[n.max_depth]
                                                                        , subsample = cv.subsample[n.subsample]
                                                                        , colsample_bytree = cv.colsample_bytree[n.colsample_bytree]
                                                                        , gamma = cv.gamma[n.gamma]
                                                                        , metrics = "rmse"
                                                        )
                                                        # , feval = QuadraticWeightedKappa
                                                        , early.stop.round = 20
                                                        , maximize = F
                                                        , print.every.n = cv.print.every.n[n]
                                                        , nrounds = cv.nrounds[n]
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
                            
                            vec.m <- c(vec.m, m)
                            vec.n <- c(vec.n, n)
                            vec.n.min_child_weight <- c(vec.n.min_child_weight, n.min_child_weight)
                            vec.n.max_depth <- c(vec.n.max_depth, n.max_depth)
                            vec.n.subsample <- c(vec.n.subsample, n.subsample)
                            vec.n.colsample_bytree <- c(vec.n.colsample_bytree, n.colsample_bytree)
                            vec.n.gamma <- c(vec.n.gamma, n.gamma)
                            vec.score <- c(vec.score, score.folds)
                            print(paste("-----------m = ", m
                                        , "; n = ", n
                                        , "; n.min_child_weight = ", n.min_child_weight
                                        , "; n.max_depth = ", n.max_depth
                                        , "; n.subsample = ", n.subsample
                                        , "; n.colsample_bytree = ", n.colsample_bytree
                                        , "; n.gamma = ", n.gamma
                                        , "; score =", score.folds))
                        }
                    }
                }
            }
        }
    }
}
dt.result <- data.table(booster = vec.m
                        , eta = vec.n
                        , min_child_weight = vec.n.min_child_weight
                        , max_depth = vec.n.max_depth
                        , subsample = vec.n.subsample
                        , colsample_bytree = vec.n.colsample_bytree
                        , gamma = vec.n.gamma
                        , score = vec.score)
dt.result
#   booster params     score
# 1:       1      1 0.6516572
# 2:       1      2 0.6554311 *
# 3:       1      3 0.6524763
# 4:       1      4 0.6504982
# 5:       1      5 0.6488586
# 6:       1      6 0.6407940
# 7:       1      7 0.6121644
# 8:       2      1 0.6074044
# 9:       2      2 0.6115734
# 10:       2      3 0.6094382
# 11:       2      4 0.6105730
# 12:       2      5 0.6106793
# 13:       2      6 0.6111796
# 14:       2      7 0.6110618
################################
## 1.3 train a xgb #############
################################
# m == 1; n == 2 produced the bset result
# reproduce it
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
m <- 1; n <- 2
# set up a score metric for folds
score.folds <- 0
pred.train <- rep(0, dim(dt.train)[1])
pred.valid <- rep(0, dim(dt.valid)[1])
pred.test <- rep(0, dim(dt.test)[1])

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
    set.seed(m * 8 + n * 64 + k * 512)
    cv.xgb.out <- xgb.train(data = dmx.train.fold
                            , booster = cv.booster[m]
                            # , objective = "reg:linear"
                            , objective = "count:poisson"
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
    pred.train <- pred.train + predict(cv.xgb.out, dmx.train)
    pred.valid <- pred.valid + predict(cv.xgb.out, dmx.valid)
    pred.test <- pred.test + predict(cv.xgb.out, x.test)
}

pred.train <- pred.train / 3
pred.valid <- pred.valid / 3
pred.test <- pred.test / 3

# pred.train <- predict(md.xgb, dmx.train)
SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
    cuts <- c(min(pred.train), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.train))
    pred <- as.integer(cut2(pred.train, cuts))
    err <- ScoreQuadraticWeightedKappa(pred, y.train, 1, 8)
    return(-err)
}
optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
optCuts

# pred.valid <- predict(md.xgb, dmx.valid)
cuts <- c(min(pred.valid), optCuts$par, max(pred.valid))
score <- ScoreQuadraticWeightedKappa(y.valid, as.integer(cut2(pred.valid, cuts)))
score
# [1] 0.6594006 poisson

## create submission file
# pred.test <- predict(md.xgb, x.test)
submission = data.table(Id = dt.test$Id)
cuts <- c(min(pred.test), optCuts$par, max(pred.test))
submission$Response = as.integer(cut2(pred.test, cuts))
table(submission$Response)
# 1    2    3    4    5    6    7    8 
# 1559 1037 1383 1609 2914 2297 3307 5659 
write.csv(submission, "submit/007_xgb_poisson_after_cv_v2.csv", row.names = FALSE) # 0.66713










