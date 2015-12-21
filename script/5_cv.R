rm(list = ls()); gc();
setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Prudential_Life_Insurance_Assessment/")
load("data/data_preprocess/dt_proprocess_combine.RData")
require(data.table)
require(caret)
############################################################################################
## 1.0 xgboost - gbtree ####################################################################
############################################################################################
require(xgboost)
require(Ckmeans.1d.dp)
x.train <- model.matrix(Response ~., dt.preprocessed.combine[isTest == 0, !c("Id", "isTest"), with = F])[, -1]
y.train <- dt.preprocessed.combine[isTest == 0]$Response - 1
dmx.train <- xgb.DMatrix(data =  x.train, label = y.train)

# x.valid1 <- model.matrix(PRED ~., dt.valid1[, !c("ACCOUNT_ID"), with = F])[, -1]
# y.valid1 <- ifelse(as.integer(dt.valid1$PRED) == 1, 0, 1)
# dmx.valid1 <- xgb.DMatrix(data =  x.valid1, label = y.valid1)

# x.valid2 <- model.matrix(PRED ~., dt.valid2[, !c("ACCOUNT_ID"), with = F])[, -1]
# y.valid2 <- ifelse(as.integer(dt.valid2$PRED) == 1, 0, 1)
# dmx.valid2 <- xgb.DMatrix(data =  x.valid2, label = y.valid2)

x.test <- model.matrix(~., dt.preprocessed.combine[isTest == 1, !c("Id", "isTest", "Response"), with = F])[, -1]

evalerror <- function(preds, dtrain){
    labels <- getinfo(dtrain, "label")
    err <- ScoreQuadraticWeightedKappa(labels,round(preds))
    
    return(list(metric = "kappa", value = err))
}

set.seed(1)
cv.xgb.out <- xgb.cv(data = dmx.train
                     , booster = "gbtree"
                     , objective = "multi:softmax"
                     , num_class = 8
                     , params = list(nthread = 8
                                     , feval = evalerror
                                     , eta = .3
                                     , max_depth = 16
                                     , subsample = 1
                                     , colsample_bytree = 1
                                     )
                     , nrounds = 500
                     , nfold = 5
                     , verbose = T)
cv.xgb.out
# train.merror.mean train.merror.std test.merror.mean test.merror.std
# 1:          0.265093         0.001707         0.457992        0.004032
# 2:          0.222824         0.002236         0.442970        0.004030
# 3:          0.196658         0.001475         0.438760        0.003786
# 4:          0.178972         0.001054         0.435071        0.002821
# 5:          0.164430         0.000798         0.433152        0.002445
# ---                                                                    
# 496:          0.000000         0.000000         0.420319        0.003892
# 497:          0.000000         0.000000         0.420319        0.003944
# 498:          0.000000         0.000000         0.420353        0.004047
# 499:          0.000000         0.000000         0.420067        0.004003
# 500:          0.000000         0.000000         0.420151        0.004034

md.xgb.out <- xgb.train(data = dmx.train
                        , booster = "gbtree"
                        , objective = "multi:softmax"
                        , num_class = 8
                        , params = list(nthread = 8
                                        , feval = evalerror
                                        , eta = .3
                                        , max_depth = 16
                                        , subsample = 1
                                        , colsample_bytree = 1
                        )
                        , nrounds = 500
                        # , nfold = 5
                        , verbose = T)

