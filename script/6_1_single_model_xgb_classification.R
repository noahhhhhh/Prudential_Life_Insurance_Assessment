rm(list = ls()); gc();
setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Prudential_Life_Insurance_Assessment/")
load("data/data_preprocess/dt_proprocess_combine.RData")
source("script/utilities/metrics.R")
source("script/utilities/preprocess.R")
require(data.table)
require(caret)
require(Metrics)
require(Hmisc)
require(caTools)
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
ls.pred.threshold.encode <- list()
ls.pred.train.encode <- list()
ls.pred.valid.encode <- list()
ls.pred.test.encode <- list()
for(r in 1:8){
    y.train.encode <- ifelse(dt.train$Response %in% 1:r, 1, 0)
    dmx.train.encode <- xgb.DMatrix(data =  x.train, label = y.train.encode)
    
    y.valid.encode <- ifelse(dt.valid$Response %in% 1:r, 1, 0)
    dmx.valid.encode <- xgb.DMatrix(data = x.valid, label = y.valid.encode)
    
    cat("training ...\n")
    print(paste("encoded reponse ==", r))
    set.seed(888)
    cv.xgb.out <- xgb.train(data = dmx.train.encode
                            , booster = "gbtree"
                            , objective = "binary:logistic"
                            , params = list(nthread = 8
                                            , eta = .025
                                            , min_child_weight = 20
                                            , max_depth = 8
                                            , subsample = .8
                                            , colsample_bytree = .8
                                            , metrics = "error"
                            )
                            , early.stop.round = 20
                            , maximize = F
                            , print.every.n = 25
                            , nrounds = 18000
                            , watchlist = list(valid = dmx.valid.encode, train = dmx.train.encode)
                            , verbose = T
    )
    cat("\npredicting on train, valid, and test encode ...\n")
    pred.train.encode <- predict(cv.xgb.out, dmx.train.encode)
    pred.valid.encode <- predict(cv.xgb.out, dmx.valid.encode)
    pred.test.encode <- predict(cv.xgb.out, x.test)
    
#     cat("optimising binary decision boundary ...\n")
#     Binfun <- function(x = .5){
#         cuts <- c(min(pred.train.encode), x, max(pred.train.encode))
#         pred <- as.integer(cut2(pred.train.encode, cuts)) - 1
#         tab <- table(pred, y.train.encode)
#         err <- 1 - sum(diag(tab)) / sum(tab)
#         return(err)
#     }
#     optCuts <- optim(.5, Binfun)
#     optCuts
#     
#     cat("apply optimised boundary on train, valid, and test encode ...\n")
#     cuts <- optCuts$par
#     pred.train.encode.op <- as.integer(cut2(pred.train.encode, cuts)) - 1
#     pred.valid.encode.op <- as.integer(cut2(pred.valid.encode, cuts)) - 1
#     pred.test.encode.op <- as.integer(cut2(pred.test.encode, cuts)) - 1
#     
#     cat("optimised train QW score: ")
#     tab.train <- table(pred.train.encode.op, y.train.encode)
#     print(1 - sum(diag(tab.train)) / sum(tab.train))
#     
#     cat("optimised vaild QW score:")
#     tab.valid <- table(pred.valid.encode.op, y.valid.encode)
#     print(1 - sum(diag(tab.valid)) / sum(tab.valid))
    
    ls.pred.train.encode[[r]] <- pred.train.encode
    ls.pred.valid.encode[[r]] <- pred.valid.encode
    ls.pred.test.encode[[r]] <- pred.test.encode
    # ls.pred.threshold.encode[[r]] <- cuts
}
cat("get the probability of each response")
dt.pred.train <- as.data.table(sapply(ls.pred.train.encode, print))
pred.train <- apply(dt.pred.train, 1, which.max)
ScoreQuadraticWeightedKappa(pred.train, y.train)
# [1] 0.5469724

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

dt.pred.valid <- as.data.table(sapply(ls.pred.valid.encode, print))
pred.valid <- apply(dt.pred.valid, 1, which.max)
ScoreQuadraticWeightedKappa(pred.valid, y.valid)
# [1] 0.5469724




