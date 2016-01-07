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

############################################################################################
## 2.0 random forest #######################################################################
############################################################################################
################################
## 2.1 train, valid, and test ##
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
## 2.2 cv rf ###################
################################
require(ranger)
dt.train.rf <- dt.train[, !c("Id", "isTest"), with = F]
# dt.train.rf[, Response := as.factor(Response)]
# mtry <- round(sqrt(dim(dt.train.rf)[2]))
mtry <- ceiling(dim(dt.train.rf)[2] / 3)

score.cv <- as.numeric()
vec.n.trees <- as.numeric()
vec.n.mtry <- as.numeric()

for (n.trees in seq(500, 5500, by = 1000)){
    for (n.mtry in seq(12, 120, by = 24)){
        set.seed(888)
        md.rf <- ranger(Response ~.
                        , data = dt.train.rf
                        , num.trees = n.trees
                        , mtry = n.mtry
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
        # [1] 0.6265719 (originally .550676; num.trees = 500, regression tree on raw features)
        # [1] 0.5286518 (originally .5286518; num.trees = 500, classification tree on raw features)
        # [1] 0.6444031 (originally .5558768; num.trees = 5000, regression tree on raw features)
        
        print(paste("-------- n.trees:", n.trees, "; mtry:", n.mtry, "; score:", score))
        
        score.cv <- c(score.cv, score)
        vec.n.trees <- c(vec.n.trees, n.trees)
        vec.n.mtry <- c(vec.n.mtry, n.mtry)
    }
}
dt.result <- data.table(num.trees = vec.n.trees, mtry = vec.n.mtry, score = score.cv)
dt.result
# num.trees mtry     score
# 1:       500   12 0.6277843
# 2:       500   36 0.6429618
# 3:       500   60 0.6402890
# 4:       500   84 0.6439057
# 5:       500  108 0.6405881
# 6:      1500   12 0.6300243
# 7:      1500   36 0.6426614
# 8:      1500   60 0.6429911
# 9:      1500   84 0.6429398
# 10:      1500  108 0.6435185
# 11:      2500   12 0.6281866
# 12:      2500   36 0.6424316
# 13:      2500   60 0.6448425
# 14:      2500   84 0.6429409
# 15:      2500  108 0.6441859
# 16:      3500   12 0.6293591
# 17:      3500   36 0.6450072 *
# 18:      3500   60 0.6443587
# 19:      3500   84 0.6432712
# 20:      3500  108 0.6429184
# 21:      4500   12 0.6296491
# 22:      4500   36 0.6445917
# 23:      4500   60 0.6436474
# 24:      4500   84 0.6441578
# 25:      4500  108 0.6433275
# 26:      5500   12 0.6295995
# 27:      5500   36 0.6442807
# 28:      5500   60 0.6438699
# 29:      5500   84 0.6432383
# 30:      5500  108 0.6437692











