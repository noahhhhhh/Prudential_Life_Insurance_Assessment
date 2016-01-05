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
cat("prepare train, valid, and test data set...\n")
set.seed(888)
ind.train <- createDataPartition(dt.preprocessed.combine[isTest == 0]$Response, p = .66, list = F)
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
cat("create a 3-folds...\n")
set.seed(888)
# create a 3 folds
folds <- createFolds(dt.train$Response, k = 3, list = F)

cat("reproduce from cv...\n")
n <- 2
n.min_child_weight <- 3
n.max_depth <- 3
n.gamma <- 3

cat("init some variables...\n")
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
    
    cat("train\n")
    for(k in 1:3){ # folds
        set.seed(n * 64 + n.min_child_weight * 128 + n.max_depth * 256 + n.gamma * 512 + k * 1024 + s * 1024)
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
        set.seed(n * 64 + n.min_child_weight * 128 + n.max_depth * 256 + n.gamma * 512 + k * 1024 + s * 1024)
        cv.xgb.out <- xgb.train(data = dmx.train.fold
                                , booster = "gbtree"
                                , objective = "count:poisson"
                                , params = list(nthread = 8
                                                , eta = .025
                                                , min_child_weight = 20 # 30 is cv best
                                                , max_depth = 8
                                                , subsample = .8
                                                , colsample_bytree = .8
                                                , gamma = .8
                                                , metrics = "rmse"
                                )
                                # , feval = QuadraticWeightedKappa
                                , early.stop.round = 20
                                , maximize = F
                                , print.every.n = 150
                                , nrounds = 15000
                                , watchlist = list(valid = dmx.valid.fold, train = dmx.train.fold)
                                , verbose = T
        )
        
        pred.train.fold.temp <- predict(cv.xgb.out, dmx.train.fold)
        pred.valid.fold.temp <- predict(cv.xgb.out, dmx.valid.fold)
        SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
            cuts <- c(min(pred.train.fold.temp), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.train.fold.temp))
            pred <- as.numeric(cut2(pred.train.fold.temp, cuts))
            err <- ScoreQuadraticWeightedKappa(pred, y.train.fold, 1, 8)
            return(-err)
        }
        optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
        optCuts
        
        cuts <- c(min(pred.valid.fold.temp), optCuts$par, max(pred.valid.fold.temp))
        score <- ScoreQuadraticWeightedKappa(y.valid.fold, as.integer(cut2(pred.valid.fold.temp, cuts)))
        print(paste("------- loop:", s, "; k:", k, "; score:", score))
        
        pred.train <- pred.train + predict(cv.xgb.out, dmx.train)
        pred.valid <- pred.valid + predict(cv.xgb.out, dmx.valid)
        pred.test <- pred.test + predict(cv.xgb.out, x.test)
    }
    pred.train <- pred.train / 3
    pred.valid <- pred.valid / 3
    pred.test <- pred.test / 3
    
    cat("optimise the cuts on pred.train...\n")
    SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
        cuts <- c(min(pred.train), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.train))
        pred <- as.integer(cut2(pred.train, cuts))
        err <- ScoreQuadraticWeightedKappa(pred, y.train, 1, 8)
        return(-err)
    }
    optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
    optCuts
    
    cat("predict the valid...\n")
    cuts.valid <- c(min(pred.valid), optCuts$par, max(pred.valid))
    pred.valid.op <- as.integer(cut2(pred.valid, cuts.valid))
    print(paste("loop", s, ": score -", ScoreQuadraticWeightedKappa(y.valid, pred.valid.op)))
    
    cat("predict the test...\n")
    cuts.test <- c(min(pred.test), optCuts$par, max(pred.test))
    pred.test.op <- as.integer(cut2(pred.test, cuts.test))
    
    cat("combine the result\n")
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
head(dt.pred.train)
head(dt.pred.valid)
head(dt.pred.test)

cat("transform the op\n")
dt.pred.valid.op <- as.data.table(sapply(ls.pred.valid.op, print))
dt.pred.test.op <- as.data.table(sapply(ls.pred.test.op, print))
head(dt.pred.valid.op)
head(dt.pred.test.op)

cat("transform optCuts\n")
dt.optCuts <- as.data.table(sapply(ls.optCuts, print))
head(dt.optCuts)

cat("median combine the preds\n")
pred.train.final <- apply(dt.pred.valid, 1, function(x) median(x))
pred.valid.final <- apply(dt.pred.valid, 1, function(x) median(x))
pred.test.final <- apply(dt.pred.test, 1, function(x) median(x))

pred.valid.final.op <- apply(dt.pred.valid.op, 1, function(x) median(x))
pred.test.final.op <- apply(dt.pred.test.op, 1, function(x) median(x))

cat("check the valid score")
score <- ScoreQuadraticWeightedKappa(y.valid, pred.valid.final.op)
score
# [1] 0.6605907 with all features (LB 0.66705)
# with no 4 group features

################################
## 1.3 submit ##################
################################
submission = data.table(Id = dt.test$Id)
submission$Response = round(pred.test.final.op)
table(submission$Response)
# 1    2    3    4    5    6    7    8 
# 1548 1019 1379 1763 2235 2912 3151 5758 
write.csv(submission, "submit/011_xgb_poisson_recv_with_all_features.csv", row.names = FALSE) # 0.66705













