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

cat("set the parameters\n")

cv.eta <- c(.001, .025, .05)
cv.nrounds <- c(18000, 15000, 12000)
cv.print.every.n <- c(250, 150, 100)

cv.min_child_weight <- c(10, 20, 30)
cv.max_depth <- c(6, 7, 8)
cv.gamma <- c(.1, .5, .8)

# set up vector n
vec.n <- as.numeric()
# set up vector n.min_child_weight
vec.n.min_child_weight <- as.numeric()
# set up vector n.max_depth
vec.n.max_depth <- as.numeric()
# set up vector n.gamma
vec.n.gamma <- as.numeric()

# set up vector score
vec.score <- as.numeric()

# setup the predictions
# train a model on folds

for(n in 1:3){ # eta; nrounds; print.every.n
    for(n.min_child_weight in 1:3){ # min_child_weight
        for(n.max_depth in 1:3){ # max_depth
            for(n.gamma in 1:3){
                # set up a score metric for folds
                score.folds <- 0
                for(k in 1:3){ # folds
                    set.seed(n * 64 + n.min_child_weight * 128 + n.max_depth * 256 + n.gamma * 512 + k * 1024)
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
                    set.seed(n * 64 + n.min_child_weight * 128 + n.max_depth * 256 + n.gamma * 512 + k * 1024)
                    cv.xgb.out <- xgb.train(data = dmx.train.fold
                                            , booster = "gbtree"
                                            , objective = "count:poisson"
                                            , params = list(nthread = 8
                                                            , eta = cv.eta[n]
                                                            , min_child_weight = cv.min_child_weight[n.min_child_weight]
                                                            , max_depth = cv.max_depth[n.max_depth]
                                                            , subsample = 1
                                                            , colsample_bytree = 1
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
                
                vec.n <- c(vec.n, n)
                vec.n.min_child_weight <- c(vec.n.min_child_weight, n.min_child_weight)
                vec.n.max_depth <- c(vec.n.max_depth, n.max_depth)
                vec.n.gamma <- c(vec.n.gamma, n.gamma)
                vec.score <- c(vec.score, score.folds)
                print(paste("----------- n = ", n
                            , "; n.min_child_weight = ", n.min_child_weight
                            , "; n.max_depth = ", n.max_depth
                            , "; n.gamma = ", n.gamma
                            , "; score =", score.folds))
            }
        }
    }
}

dt.result <- data.table(eta = vec.n
                        , min_child_weight = vec.n.min_child_weight
                        , max_depth = vec.n.max_depth
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

# with all features
# eta   min_child_weight max_depth gamma     score
# 1:   1                3         1     2 0.6487236
# 2:   1                3         1     3 0.6489012
# 3:   3                2         1     3 0.6491869
# 4:   1                3         1     1 0.6498435
# 5:   1                2         1     1 0.6501650
# 6:   2                2         3     2 0.6504955
# 7:   1                2         2     2 0.6504981
# 8:   1                1         2     1 0.6505547
# 9:   2                2         3     1 0.6506218
# 10:   3                1         3     2 0.6506946
# 11:   3                3         2     2 0.6507400
# 12:   2                2         3     3 0.6507453
# 13:   3                3         3     2 0.6509515
# 14:   2                1         2     3 0.6510411
# 15:   2                1         2     1 0.6510937
# 16:   1                2         3     2 0.6512045
# 17:   2                1         2     2 0.6512776
# 18:   3                1         3     1 0.6513925
# 19:   3                1         1     2 0.6514067
# 20:   1                1         2     2 0.6514474
# 21:   1                1         3     2 0.6514654
# 22:   2                3         3     2 0.6515351
# 23:   1                2         1     2 0.6515469
# 24:   1                2         1     3 0.6515649
# 25:   1                2         3     1 0.6515853
# 26:   3                3         3     1 0.6516330
# 27:   3                1         2     1 0.6516345
# 28:   3                2         2     2 0.6516771
# 29:   2                1         1     2 0.6517002
# 30:   3                1         1     1 0.6517058
# 31:   3                1         1     3 0.6517366
# 32:   2                3         1     2 0.6517746
# 33:   2                2         2     3 0.6519260
# 34:   1                1         1     3 0.6520129
# 35:   3                2         3     2 0.6520683
# 36:   2                3         2     3 0.6520738
# 37:   2                3         2     1 0.6520801
# 38:   3                2         3     3 0.6521448
# 39:   3                1         2     3 0.6521606
# 40:   1                2         2     3 0.6521751
# 41:   1                1         3     3 0.6522166
# 42:   1                3         3     1 0.6523178
# 43:   3                1         3     3 0.6523364
# 44:   1                3         3     2 0.6523507
# 45:   1                2         2     1 0.6523593
# 46:   1                3         2     1 0.6523815
# 47:   2                1         3     1 0.6524081
# 48:   3                2         1     1 0.6524425
# 49:   1                2         3     3 0.6525026
# 50:   2                2         2     2 0.6525679
# 51:   3                2         3     1 0.6526211
# 52:   2                2         1     3 0.6526399
# 53:   1                3         2     3 0.6526462
# 54:   2                1         3     2 0.6526673
# 55:   1                3         3     3 0.6526840
# 56:   1                1         1     2 0.6527517
# 57:   3                3         2     1 0.6528601
# 58:   3                2         2     1 0.6529017
# 59:   3                3         1     3 0.6529732
# 60:   3                3         2     3 0.6530338
# 61:   1                3         2     2 0.6530449
# 62:   2                2         1     1 0.6531733
# 63:   3                2         2     3 0.6532243
# 64:   1                1         3     1 0.6532718
# 65:   1                1         2     3 0.6533071
# 66:   2                1         1     3 0.6533249
# 67:   3                3         1     1 0.6533615
# 68:   3                3         1     2 0.6534911
# 69:   2                1         3     3 0.6535052
# 70:   3                3         3     3 0.6535182
# 71:   3                1         2     2 0.6535676
# 72:   2                1         1     1 0.6536791
# 73:   3                2         1     2 0.6538036
# 74:   2                3         2     2 0.6539990
# 75:   2                3         1     1 0.6541206
# 76:   2                3         1     3 0.6542277
# 77:   1                1         1     1 0.6544564
# 78:   2                3         3     1 0.6548723
# 79:   2                2         1     2 0.6550858
# 80:   2                2         2     1 0.6552995
# 81:   2                3         3     3 0.6553574

################################
## 1.3 train a xgb #############
################################
load("data/data_meta/dt_train_valid_test_with_rf_meta.RData")
x.train <- model.matrix(Response ~., dt.train[, !c("Id", "isTest"), with = F])[, -1]
y.train <- dt.train$Response
dmx.train <- xgb.DMatrix(data =  x.train, label = y.train)

x.valid <- model.matrix(Response ~., dt.valid[, !c("Id", "isTest"), with = F])[, -1]
y.valid <- dt.valid$Response
dmx.valid <- xgb.DMatrix(data =  x.valid, label = y.valid)

x.test <- model.matrix(~., dt.test[, !c("Id", "isTest", "Response"), with = F])[, -1]

# m == 1; n == 2 produced the bset result
# reproduce it
cat("cross validate xgb\n")
set.seed(888)
# create a 3 folds
folds <- createFolds(dt.train$Response, k = 3, list = F)

cat("set the parameters\n")
# 1: m
cv.booster <- c("gbtree", "gblinear")
# 2: n
cv.eta <- c(.001, .025, .05, .075, .1, .2, .5)
cv.nrounds <- c(12000, 8000, 6000, 5000, 4000, 3000, 2500)
cv.print.every.n <- c(250, 150, 100, 75, 50, 40, 30)
# 3: n
cv.min_child_weight <- c(10, 20, 20, 30, 30, 40, 40)
cv.max_depth <- rep(8, 7)
cv.subsample <- c(.9, .8, .8, .7, .7, .6, .6)
cv.colsample_bytree <- c(.8, .8, .7, .7, .6, .6, .5)

# reproduce with m = 1 and n = 2
cat("start training\n")
m <- 1; n <- 2
ls.pred.train <- list()
ls.pred.valid <- list()
ls.pred.test <- list()
ls.pred.valid.op <- list()
ls.pred.test.op <- list()
ls.optCuts <- list()
for(s in 1:15){
    # set up a score metric for folds
    pred.train <- rep(0, dim(dt.train)[1])
    pred.valid <- rep(0, dim(dt.valid)[1])
    pred.test <- rep(0, dim(dt.test)[1])
    
    for(k in 1:3){ # folds
        set.seed(m * 8 + n * 64 + k * 512 + s * 1024)
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
        set.seed(m * 8 + n * 64 + k * 512 + s * 1024)
        cv.xgb.out <- xgb.train(data = dmx.train.fold
                                , booster = cv.booster[m] # gbtree
                                , objective = "count:poisson"
                                , params = list(nthread = 8
                                                , eta = cv.eta[n] # .025
                                                , min_child_weight = cv.min_child_weight[n] # 10
                                                , max_depth = cv.max_depth[n] # 8
                                                , subsample = cv.subsample[n] # .8
                                                , colsample_bytree = cv.colsample_bytree[n] # .8
                                                , metrics = "rmse"
                                )
                                , early.stop.round = 20
                                , maximize = F
                                , print.every.n = cv.print.every.n[n] # 150
                                , nrounds = cv.nrounds[n] # 8000
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
    
    # optimise the cuts on pred.train
    SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
        cuts <- c(min(pred.train), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.train))
        pred <- as.integer(cut2(pred.train, cuts))
        err <- ScoreQuadraticWeightedKappa(pred, y.train, 1, 8)
        return(-err)
    }
    optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
    optCuts
    
    # predict the valid
    cuts.valid <- c(min(pred.valid), optCuts$par, max(pred.valid))
    pred.valid.op <- as.integer(cut2(pred.valid, cuts.valid))
    print(paste("loop", s, ": score -", ScoreQuadraticWeightedKappa(y.valid, pred.valid.op)))
    # predict the test
    cuts.test <- c(min(pred.test), optCuts$par, max(pred.test))
    pred.test.op <- as.integer(cut2(pred.test, cuts.test))
    
    # combine the result
    ls.pred.train[[s]] <- pred.train
    ls.pred.valid[[s]] <- pred.valid
    ls.pred.test[[s]] <- pred.test
    
    ls.pred.valid.op[[s]] <- pred.valid.op
    ls.pred.test.op[[s]] <- pred.test.op
    
    ls.optCuts[[s]] <- optCuts$par
}
cat("transform the train, valid, and test\n")
dt.pred.train <- as.data.table(sapply(ls.pred.train, print))
dt.pred.valid <- as.data.table(sapply(ls.pred.valid, print))
dt.pred.test <- as.data.table(sapply(ls.pred.test, print))
cat("transform the op\n")
dt.pred.valid.op <- as.data.table(sapply(ls.pred.valid.op, print))
dt.pred.test.op <- as.data.table(sapply(ls.pred.test.op, print))
cat("transform optCuts\n")
dt.optCuts <- as.data.table(sapply(ls.optCuts, print))

dt.pred.train
dt.pred.valid
dt.pred.test

dt.pred.valid.op
dt.pred.test.op

dt.optCuts

cat("median combine the preds\n")
pred.train.final <- apply(dt.pred.valid, 1, function(x) median(x))
pred.valid.final <- apply(dt.pred.valid, 1, function(x) median(x))
pred.test.final <- apply(dt.pred.test, 1, function(x) median(x))

pred.valid.final.op <- apply(dt.pred.valid.op, 1, function(x) median(x))
pred.test.final.op <- apply(dt.pred.test.op, 1, function(x) median(x))

cat("check the score")
score <- ScoreQuadraticWeightedKappa(y.valid, pred.valid.final.op)
score
# highest SQW score!
# [1] 0.6601923 * this is highest, .8
# [1] 0.5906142 apply + median + opCuts.final
# [1] 0.6602611 col sample .6 with loop = 15 - overfitted (0.66785)
# [1] 0.6620329 adding age, wt, ht, BMI group - overfitted (0.66587)


## create submission file with poisson
# pred.test <- predict(md.xgb, x.test)
submission = data.table(Id = dt.test$Id)
cuts <- c(min(pred.test), optCuts$par, max(pred.test))
submission$Response = as.integer(cut2(pred.test, cuts))
table(submission$Response)
# 1    2    3    4    5    6    7    8 
# 1559 1037 1383 1609 2914 2297 3307 5659 
write.csv(submission, "submit/007_xgb_poisson_after_cv_v2.csv", row.names = FALSE) # 0.66713

## create submission file with s = 15, with rounding
submission = data.table(Id = dt.test$Id)
submission$Response = round(pred.test.final / 15)
table(submission$Response)
# 1    2    3    4    5    6    7    8 
# 1714  935 1504 1694 2258 2672 3333 5655
write.csv(submission, "submit/008_xgb_poisson_with_mult_sample_with_rounding_ensemble.csv", row.names = FALSE) # 0.66819 *

## create submission file with col sample = .6, with s = 15, with median combine
submission = data.table(Id = dt.test$Id)
submission$Response = round(pred.test.final.op)
table(submission$Response)
# 1    2    3    4    5    6    7    8 
# 1705  881 1625 1570 2437 2563 3380 5604 
write.csv(submission, "submit/009_xgb_poisson_with_colsample_06_with_multi_loop_15_with_median_combine.csv", row.names = FALSE) # 0.66785

## create submission file with 4 groups features, with s = 15, with median combine
submission = data.table(Id = dt.test$Id)
submission$Response = round(pred.test.final.op)
table(submission$Response)
# 1    2    3    4    5    6    7    8 
# 1656 1012 1512 1631 2361 2574 3271 5748 
write.csv(submission, "submit/010_xgb_poisson_with_4_groups_with_multi_loop_15_with_median_combine.csv", row.names = FALSE) # 0.66587

############################################################################################
## 2.0 random forest #######################################################################
############################################################################################
################################
## 2.1 train, valid, and test ##
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
## 2.2 train a model ###########
################################
require(doParallel)
require(randomForest)
cores <- detectCores()
cl <- makeCluster(cores)
registerDoParallel(cl)

set.seed(1)
md.rf <- foreach(ntree = rep(250, 8)
                 , .combine = combine
                 , .packages = "randomForest") %dopar% 
    randomForest(x = x.train
                 , y = y.train
                 , ntree = ntree
                 , mtry = floor(sqrt(ncol(x.train)))
                 , replace = T
                 , nodesize = 100
                 , importance = T
                 , keep.forest = T
    )
stopCluster(cl)

pred.rf.train <- predict(md.rf, newdata = x.train)
pred.rf.valid <- predict(md.rf, newdata = x.valid)
pred.rf.test <- predict(md.rf, newdata = x.test)

ScoreQuadraticWeightedKappa(y.train, as.integer(cut2(pred.rf.train, c(-Inf, seq(1.5, 7.5, by = 1), Inf))))
# [1] 0.5467729

# optimise the cuts on pred.train
SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
    cuts <- c(-Inf, x[1], x[2], x[3], x[4], x[5], x[6], x[7], Inf)
    pred <- as.integer(cut2(pred.rf.train, cuts))
    err <- ScoreQuadraticWeightedKappa(pred, y.train, 1, 8)
    return(-err)
}
optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
optCuts

# QWK score of the train
cuts.train <- c(min(pred.rf.train), optCuts$par, max(pred.rf.train))
pred.train.op <- as.integer(cut2(pred.rf.train, cuts.train))
ScoreQuadraticWeightedKappa(y.train, pred.train.op)
# [1] 0.6804625

# QWK score of the valid
cuts.valid <- c(min(pred.rf.valid), optCuts$par, max(pred.rf.valid))
pred.valid.op <- as.integer(cut2(pred.rf.valid, cuts.valid))
ScoreQuadraticWeightedKappa(y.valid, pred.valid.op)
# [1] 0.620452

# predict test
cuts.test <- c(min(pred.rf.test), optCuts$par, max(pred.rf.test))
pred.test.op <- as.integer(cut2(pred.rf.test, cuts.test))

# adding a meta feature
dt.train[, rf_meta := pred.train.op]
dt.valid[, rf_meta := pred.valid.op]
dt.test[, rf_meta := pred.test.op]
save(dt.train, dt.valid, dt.test, file = "data/data_meta/dt_train_valid_test_with_rf_meta.RData") # getting worse

############################################################################################
## 3.0 random forest #######################################################################
############################################################################################
################################
## 3.1 train, valid, and test ##
################################
cat("prepare train, valid, and test data set\n")
set.seed(888)
ind.train <- createDataPartition(dt.preprocessed.combine[isTest == 0]$Response, p = .9, list = F)
dt.train <- dt.preprocessed.combine[isTest == 0][ind.train][, !c("Id", "isTest"), with = F]
dt.valid <- dt.preprocessed.combine[isTest == 0][-ind.train][, !c("Id", "isTest"), with = F]
dt.test <- dt.preprocessed.combine[isTest == 1]
dim(dt.train); dim(dt.valid); dim(dt.test)

################################
## 3.2 train a model ###########
################################
md.pr <- glm(Response ~., family = "gaussian", data = dt.train)
pred.pr.train <- predict(md.pr, dt.train)
pred.pr.valid <- predict(md.pr, dt.valid)

ScoreQuadraticWeightedKappa(dt.train$Response, as.integer(cut2(pred.pr.train, c(-Inf, seq(1.5, 7.5, by = 1), Inf))))
# 0.6671775


# optimise the cuts on pred.train
SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
    cuts <- c(-Inf, x[1], x[2], x[3], x[4], x[5], x[6], x[7], Inf)
    pred <- as.integer(cut2(pred.pr.train, cuts))
    err <- ScoreQuadraticWeightedKappa(pred, dt.train$Response, 1, 8)
    return(-err)
}
optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
optCuts

# QWK score of the train
cuts.train <- c(min(pred.pr.train), optCuts$par, max(pred.pr.train))
pred.train.op <- as.integer(cut2(pred.pr.train, cuts.train))
ScoreQuadraticWeightedKappa(dt.train$Response, pred.train.op)
# [1] 0.7108755

# QWK score of the valid
cuts.valid <- c(min(pred.pr.valid), optCuts$par, max(pred.pr.valid))
pred.valid.op <- as.integer(cut2(pred.pr.valid, cuts.valid))
ScoreQuadraticWeightedKappa(dt.valid$Response, pred.valid.op)
# [1] 0.6376342











