rm(list = ls()); gc();
setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Prudential_Life_Insurance_Assessment/")
load("data/data_preprocess/dt_proprocess_combine.RData")
source("script/utilities/metrics.R")
source("script/utilities/preprocess.R")
require(data.table)
require(caret)
require(Metrics)
require(Hmisc)
require(xgboost)
require(Ckmeans.1d.dp)
require(glmnet)
require(ranger)
############################################################################################
## 1.0 train, valid, and test ##############################################################
############################################################################################
################################
## 1.1 train, valid, and test ##
################################
cat("prepare train, valid, and test data set...\n")
# set.seed(999)
# ind.train <- createDataPartition(dt.preprocessed.combine[isTest == 0]$Response, p = .8, list = F) # remember to change it to .66
load("model/lr.RData")
head(ind.train)
dt.train <- dt.preprocessed.combine[isTest == 0][ind.train]
dt.train[, pred.lr := response.train]
dt.valid <- dt.preprocessed.combine[isTest == 0][-ind.train]
dt.valid[, pred.lr := response.valid]
dt.test <- dt.preprocessed.combine[isTest == 1]
dt.test[, pred.lr := response.test]
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
m <- 1; n <- 2
cat("creating 3 folds ...\n")
set.seed(888)
# create a 4 folds
folds <- createFolds(dt.train$Response, k = 3, list = F)

# reproduce with m = 1 and n = 2
cat("initiating list variables ...\n")
ls.pred.train.rf <- list()
ls.pred.valid.rf <- list()
ls.pred.test.rf <- list()
ls.pred.train.xgb <- list()
ls.pred.valid.xgb <- list()
ls.pred.test.xgb <- list()
ls.pred.valid.op <- list()
ls.pred.test.op <- list()
ls.optCuts <- list()
cat("init vector variables ...\n")
pred.train.rf <- rep(0, dim(dt.train)[1])
pred.valid.rf <- rep(0, dim(dt.valid)[1])
pred.test.rf <- rep(0, dim(dt.test)[1])
pred.train.xgb <- rep(0, dim(dt.train)[1])
pred.valid.xgb <- rep(0, dim(dt.valid)[1])
pred.test.xgb <- rep(0, dim(dt.test)[1])

cat("define objective function ...\n")
evalerror <- function(preds, dtrain){
    labels <- getinfo(dtrain, "label")
    err <- ScoreQuadraticWeightedKappa(labels,round(preds))
    
    return(list(metric = "kappa", value = err))
}

cat("training ...\n")
dt.train.rf <- dt.train
dt.valid.rf <- dt.valid
dt.test.rf <- dt.test
dt.train.xgb <- dt.train
dt.valid.xgb <- dt.valid
dt.test.xgb <- dt.test
for(s in 1:5){
    cat("train, valid, test, and sample ...\n")
    set.seed(s * 2048)
#     sp1.xgb <- sample(nrow(dt.train), nrow(dt.train), replace = T)
#     sp2.rf <- setdiff(1:nrow(dt.train), sp1.xgb)
    ind.xgb <- createDataPartition(dt.train$Response, p = .67, list = F)
    ind.rf <- setdiff(1:nrow(dt.train), ind.xgb)
    cat("rf: prepare dt.train.rf ...\n")
    if(s == 1){
        dt.train.rf <- dt.train.rf
        dt.valid.rf <- dt.valid
        dt.test.rf <- dt.test
    } else{
        dt.train.rf[, pred.xgb := pred.train.xgb]
        dt.valid.rf[, pred.xgb := pred.valid.xgb]
        dt.test.rf[, pred.xgb := pred.test.xgb]
    }
    ########
    ## rf ##
    ########
    cat("rf: train rf ...\n")
    set.seed(m * 8 + n * 64 + s * 1024)
    md.rf <- ranger(Response ~.
                    , data = dt.train.rf[, !c("Id", "isTest"), with = F][ind.rf, ]
                    , num.trees = 3500
                    , mtry = 36
                    , replace = F
                    , importance = "impurity"
                    , write.forest = T
                    , min.node.size = 20
                    , num.threads = 8
                    , verbose = T
    )
    cat("rf: predict with rf ...\n")
    pred.train.rf <- predictions(predict(md.rf, dt.train.rf))
    pred.valid.rf <- predictions(predict(md.rf, dt.valid.rf))
    pred.test.rf <- predictions(predict(md.rf, dt.test.rf))
    cat("rf: store individual rf model ...\n")
    ls.pred.train.rf[[s]] <- pred.train.rf
    ls.pred.valid.rf[[s]] <- pred.valid.rf
    ls.pred.test.rf[[s]] <- pred.test.rf
    cat("rf: score...\n")
    print(ScoreQuadraticWeightedKappa(as.integer(pred.valid.rf), y.valid))
    # 0.5276916
    # 0.5724534
    
    #########
    ## xgb ##
    #########
    cat("xgb: prepare dmx.train ...\n")
    dt.train.xgb[, pred.rf := pred.train.rf]
    x.train.xgb.ind <- data.matrix(dt.train.xgb[, !c("Id", "isTest", "Response"), with = F][ind.xgb, ])
    x.train <- data.matrix(dt.train.xgb[, !c("Id", "isTest"), with = F])[, -1]
    y.train.xgb <- dt.train.xgb[ind.xgb]$Response
    y.train <- dt.train.xgb$Response
    dmx.train.xgb <- xgb.DMatrix(data =  x.train.xgb.ind, label = y.train.xgb)
    dmx.train <- xgb.DMatrix(data =  x.train, label = y.train)
    
    dt.valid.xgb[, pred.rf := pred.valid.rf]
    x.valid <- data.matrix(dt.valid.xgb[, !c("Id", "isTest", "Response"), with = F])
    y.valid <- dt.valid$Response
    dmx.valid <- xgb.DMatrix(data =  x.valid, label = y.valid)
    
    dt.test.xgb[,pred.rf := pred.test.rf]
    x.test <- model.matrix(~., dt.test.xgb[, !c("Id", "isTest", "Response"), with = F])[, -1]
    
    cat("xgb: train xgb ...\n")
    set.seed(m * 8 + n * 64 + s * 1024)
    objectives <- c("count:poisson", "reg:linear") ################# pending to use") ################# 
    set.seed(m * 8 + n * 64 + s * 1024)
    cv.xgb.out <- xgb.train(data = dmx.train.xgb
                            , booster = "gbtree"
                            , objective = "count:poisson"
                            , params = list(nthread = 8
                                            , eta = .025
                                            , min_child_weight = 100
                                            , max_depth = 8
                                            , subsample = .8
                                            , colsample_bytree = .8
                                            , metrics = "rmse"
                            )
                            # , feval = evalerror #
                            , early.stop.round = 100
                            , maximize = F
                            # , maximize = T #
                            , print.every.n = 150
                            , nrounds = 18000
                            , watchlist = list(valid = dmx.valid, train = dmx.train)
                            , verbose = T
    )
    cat("xgb: predict with xgb ...\n")
    pred.train.xgb <- predict(cv.xgb.out, dmx.train)
    pred.valid.xgb <- predict(cv.xgb.out, dmx.valid)
    pred.test.xgb <- predict(cv.xgb.out, x.test)
    cat("xgb: store individual xgb model ...\n")
    ls.pred.train.xgb[[s]] <- pred.train.xgb
    ls.pred.valid.xgb[[s]] <- pred.valid.xgb
    ls.pred.test.xgb[[s]] <- pred.test.xgb
    cat("xgb: score...\n")
    print(ScoreQuadraticWeightedKappa(as.integer(pred.valid.xgb), y.valid))
    # 0.5935143
    # 0.5598421
}

cat("transform the train, valid, and test\n")
dt.pred.train.rf <- as.data.table(sapply(ls.pred.train.rf, print))
dt.pred.valid.rf <- as.data.table(sapply(ls.pred.valid.rf, print))
dt.pred.test.rf <- as.data.table(sapply(ls.pred.test.rf, print))
dt.pred.train.xgb <- as.data.table(sapply(ls.pred.train.xgb, print))
dt.pred.valid.xgb <- as.data.table(sapply(ls.pred.valid.xgb, print))
dt.pred.test.xgb <- as.data.table(sapply(ls.pred.test.xgb, print))

cat("blend the predictions")
dt.pred.train.rf.xgb <- (dt.pred.train.xgb[, 1, with = F] * 0.5935143 + dt.pred.train.rf[, 1, with = F] * 0.5276916) / (0.5935143 + 0.5276916)
pred.train.rf.xgb <- dt.pred.train.rf.xgb$V1
dt.pred.valid.rf.xgb <- (dt.pred.valid.xgb[, 1, with = F] * 0.5935143 + dt.pred.valid.rf[, 1, with = F] * 0.5276916) / (0.5935143 + 0.5276916)
pred.valid.rf.xgb <- dt.pred.valid.rf.xgb$V1
dt.pred.test.rf.xgb <- (dt.pred.test.xgb[, 1, with = F] * 0.5935143 + dt.pred.test.rf[, 1, with = F] * 0.5276916) / (0.5935143 + 0.5276916)
pred.test.rf.xgb <- dt.pred.test.rf.xgb$V1

cat("optimising the cuts on pred.train ...\n")
set.seed(m * 8 + n * 64 + s * 1024)
trainForOpt <- sample(length(pred.train.rf.xgb), length(pred.train.rf.xgb) * .8)
pred.train.forOpt <- pred.train.rf.xgb[trainForOpt]
SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
    cuts <- c(min(pred.train.forOpt), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.train.forOpt))
    pred <- as.integer(cut2(pred.train.forOpt, cuts))
    err <- ScoreQuadraticWeightedKappa(pred, y.train[trainForOpt], 1, 8)
    return(-err)
}
optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
optCuts

cat("applying optCuts on train ...\n")
cuts.train <- c(min(pred.train.rf.xgb), optCuts$par, max(pred.train.rf.xgb))
pred.train.op <- as.integer(cut2(pred.train.rf.xgb, cuts.train))
print(paste("train score -", ScoreQuadraticWeightedKappa(y.train, pred.train.op)))

cat("applying optCuts on valid ...\n")
cuts.valid <- c(min(pred.valid.rf.xgb), optCuts$par, max(pred.valid.rf.xgb))
pred.valid.op <- as.integer(cut2(pred.valid.rf.xgb, cuts.valid))
print(paste("valid score -", ScoreQuadraticWeightedKappa(y.valid, pred.valid.op)))

# cat("median combine the preds\n")
# pred.train.final <- apply(dt.pred.train, 1, function(x) mean(x))
# pred.valid.final <- apply(dt.pred.valid, 1, function(x) mean(x))
# pred.test.final <- apply(dt.pred.test, 1, function(x) mean(x))
# 
# pred.valid.final.op <- apply(dt.pred.valid.op, 1, function(x) median(x))
# pred.test.final.op <- apply(dt.pred.test.op, 1, function(x) median(x))

cat("median combine the opCuts")
opCuts.final <- apply(dt.optCuts, 1, function(x) mean(x))

# cat("apply opCuts on pred.valid.final")
# cuts.valid.final <- c(min(pred.valid.final), opCuts.final, max(pred.valid.final))
# pred.valid.final.op <- as.integer(pred.valid.final, opCuts.final)

cat("check the score")
score <- ScoreQuadraticWeightedKappa(y.valid, round(pred.valid.final.op))
score

################################
## 1.3 submit ##################
################################
submission = data.table(Id = dt.test$Id)
submission$Response = round(pred.test.final.op)
table(submission$Response)
write.csv(submission, "submit/035_xgb_with_regression_features.csv", row.names = FALSE) # 0.6561871 (LB 0.67183)





