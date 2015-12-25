rm(list = ls()); gc();
setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Prudential_Life_Insurance_Assessment/")
load("data/data_preprocess/dt_proprocess_combine.RData")
source("script/utilities/metrics.R")
require(data.table)
require(caret)
require(Metrics)
############################################################################################
## 1.0 xgboost - gbtree ####################################################################
############################################################################################
require(xgboost)
require(Ckmeans.1d.dp)
cat("prepare train, valid, and test data set\n")
col.impute_median <- names(dt.preprocessed.combine)[grep("Impute_Median", names(dt.preprocessed.combine))]
col.impute_num <- names(dt.preprocessed.combine)[grep("Impute_1|Impute_2016", names(dt.preprocessed.combine))]
# try impute with mean/median
dt.preprocessed.combine <- dt.preprocessed.combine[, !col.impute_num, with = F]

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

# x.valid2 <- model.matrix(PRED ~., dt.valid2[, !c("ACCOUNT_ID"), with = F])[, -1]
# y.valid2 <- ifelse(as.integer(dt.valid2$PRED) == 1, 0, 1)
# dmx.valid2 <- xgb.DMatrix(data =  x.valid2, label = y.valid2)

x.test <- model.matrix(~., dt.preprocessed.combine[isTest == 1, !c("Id", "isTest", "Response"), with = F])[, -1]

cat("cross validate xgb\n")
set.seed(1)
cv.xgb.out <- xgb.cv(data = dmx.train
                     , booster = "gbtree"
                     , objective = "reg:linear"
                     , params = list(nthread = 8
                                     , eta = .025
                                     , max_depth = 16
                                     , subsample = 1
                                     , colsample_bytree = 1
                                     )
                     , metricds = "rsme"
                     , early.stop.round = 20
                     , maximize = F
                     , nrounds = 230
                     , nfold = 3
                     , watchlist = list(train = dmx.train, valid = dmx.valid)
                     , verbose = T
                     , prediction = T
                     )
cv.xgb.out$dt
# train.rmse.mean train.rmse.std test.rmse.mean test.rmse.std
# 1:        5.566481       0.002367       5.570122      0.005044
# 2:        5.441065       0.002204       5.448319      0.004888
# 3:        5.318805       0.002026       5.330019      0.004783
# 4:        5.199889       0.001838       5.214973      0.004599
# 5:        5.084187       0.001939       5.103412      0.004409
# ---                                                            
# 226:        0.753447       0.024109       1.882757      0.015178
# 227:        0.752108       0.024230       1.882771      0.015226
# 228:        0.750607       0.024336       1.882764      0.015289
# 229:        0.748594       0.024004       1.882736      0.015280
# 230:        0.746942       0.023797       1.882719      0.015299

# Quadratic Weighted Kappa
ScoreQuadraticWeightedKappa(y.train,round(cv.xgb.out$pred))
# [1] 0.5921439

SQWKfun = function(x = seq(1.5, 7.5, by = 1)) {
    cuts = c(min(cv.xgb.out$pred), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(cv.xgb.out$pred))
    pred = as.numeric(Hmisc::cut2(cv.xgb.out$pred, cuts))
    err = Metrics::ScoreQuadraticWeightedKappa(pred, y.train, 1, 8)
    return(-err)
}
optCuts = optim(seq(1.5, 7.5, by = 1), SQWKfun)
optCuts
# $par
# [1] 2.782926 4.071472 3.490766 4.830848 5.581213 6.645683 6.074274
# 
# $value
# [1] -0.6322359
# 
# $counts
# function gradient 
# 292       NA 
# 
# $convergence
# [1] 0
# 
# $message
# NULL

# training QW Kappa
pred.train <- as.numeric(Hmisc::cut2(cv.xgb.out$pred, c(-Inf, optCuts$par, Inf)))
ScoreQuadraticWeightedKappa(pred.train, y.train)
# [1] 0.6322359

cat("train xgb\n")
set.seed(1)
pred.valid <- rep(0, length(y.valid))
for (i in 1:1){
    set.seed(i * 1024)
    md.xgb.out <- xgb.train(data = dmx.train
                            , booster = "gbtree"
                            , objective = "reg:linear"
                            , params = list(nthread = 8
                                            , eta = .025
                                            , max_depth = 16
                                            , subsample = 1
                                            , colsample_bytree = 1
                            )
                            , metricds = "rsme"
                            , early.stop.round = 20
                            , maximize = F
                            , nrounds = 297
                            # , nfold = 3
                            , watchlist = list(train = dmx.train, valid = dmx.valid)
                            , verbose = T
                            # , prediction = T
    )
    pred.valid <- pred.valid + predict(md.xgb.out, x.valid)
}

ScoreQuadraticWeightedKappa(y.valid, round(pred.valid))
# [1] 0.6148074

cat("apply the optCuts\n")
pred.valid <- as.numeric(Hmisc::cut2(pred.valid, c(-Inf, optCuts$par, Inf)))
ScoreQuadraticWeightedKappa(pred.valid, y.valid)
table(pred.valid)
# 1    2    3    4    5    6    7    8 
# 457  375  371  623  761  574  761 2015

cat("submit")
pred.test <- predict(md.xgb.out, x.test)
pred.test <- as.integer(Hmisc::cut2(pred.test, c(-Inf, optCuts$par, Inf)))
table(pred.test)
# 1    2    3    4    5    6    7    8 
# 1557 1197 1151 2114 2569 1992 2465 6720 
dt.submit = data.table(Id = dt.test$Id)
dt.submit[, Response := pred.test]
write.csv(dt.submit, "submit/003_init_xgb_with_optim.csv", row.names = FALSE)


