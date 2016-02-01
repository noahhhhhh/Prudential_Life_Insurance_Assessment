rm(list = ls()); gc();
setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Prudential_Life_Insurance_Assessment/")
load("data/data_preprocess/dt_proprocess_combine.RData")
source("script/utilities/metrics.R")
source("script/utilities/preprocess.R")
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
# m == 1; n == 2 produced the bset result
m <- 1; n <- 2

cat("creating 3 folds ...\n")
set.seed(888)
# create a 4 folds
folds <- createFolds(dt.train$Response, k = 3, list = F)

# reproduce with m = 1 and n = 2
cat("initiating variables ...\n")

ls.pred.train <- list()
ls.pred.valid <- list()
ls.pred.test <- list()
ls.pred.valid.op <- list()
ls.pred.test.op <- list()
ls.optCuts <- list()

cat("training ...\n")
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
                                , booster = "gbtree"
                                , objective = "count:poisson"
                                # , objective = "reg:linear"
                                , params = list(nthread = 8
                                                , eta = .025
                                                , min_child_weight = 20
                                                , max_depth = 8
                                                , subsample = .8
                                                , colsample_bytree = .8
                                                , metrics = "rmse"
                                )
                                , early.stop.round = 100
                                , maximize = F
                                , print.every.n = 150
                                , nrounds = 18000
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
    
    cat("optimising the cuts on pred.train ...\n")
    SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
        cuts <- c(min(pred.train), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.train))
        pred <- as.integer(cut2(pred.train, cuts))
        err <- ScoreQuadraticWeightedKappa(pred, y.train, 1, 8)
        return(-err)
    }
    optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
    optCuts
#     for(i in 1:8){
#         cuts <- as.numeric()
#         pred <- as.integer()
#         offset <- 0
#         y.pred <- as.numeric()
#         y <- as.numeric()
#         err <- 0
#         offsets <- as.numeric()
#         
#         offsetFun <- function(x = .1){
#             cuts <- c(min(pred.train), seq(1.5, 7.5, by = 1), max(pred.train))
#             pred <- as.integer(cut2(pred.train, cuts))
#             offset <- x
#             
#             pred[pred == i] <- pred[pred == i] + offset
#             y.pred <- as.integer(cut2(y.pred, cuts))
#             
#             err <- ScoreQuadraticWeightedKappa(pred, y.train)
#             
#             return(-err)
#         }
#         optOffset <- optim(.1, offsetFun, method = "Brent", lower = -8, upper = 8)
#         offsets[i] <- optOffset$par
#         
#     }
#     
#     cuts <- c(min(pred.train), seq(1.5, 7.5, by = 1), max(pred.train))
#     pred.valid.op <- as.integer(cut2(pred.valid, cuts))
#     pred.test.op <- as.integer(cut2(pred.test, cuts))
#     for(j in 1:8){
#         pred.valid[pred.valid.op == j] <- pred.valid[pred.valid.op == j] + offsets[j]
#         pred.test[pred.test.op == j] <- pred.test[pred.test.op == j] + offset[j]
#     }
#     pred.valid.op <- as.integer(cut2(pred.valid, cuts))
#     pred.test.op <- as.integer(cut2(pred.test, cuts))
#     print(paste("loop", s, ": valid score -", ScoreQuadraticWeightedKappa(y.valid, pred.valid.op)))
#     
    cat("applying optCuts on valid ...\n")
    cuts.valid <- c(min(pred.valid), optCuts$par, max(pred.valid))
    pred.valid.op <- as.integer(cut2(pred.valid, cuts.valid))
    print(paste("loop", s, ": valid score -", ScoreQuadraticWeightedKappa(y.valid, pred.valid.op)))
    # [1] "loop 1 : valid score - 0.662693668247028"
    
    cat("applying optCuts on test ...\n")
    cuts.test <- c(min(pred.test), optCuts$par, max(pred.test))
    pred.test.op <- as.integer(cut2(pred.test, cuts.test))
    
    cat("combining the optimised predictions ...\n")
    ls.pred.train[[s]] <- pred.train
    ls.pred.valid[[s]] <- pred.valid
    ls.pred.test[[s]] <- pred.test
    
    ls.pred.valid.op[[s]] <- pred.valid.op
    ls.pred.test.op[[s]] <- pred.test.op
    
    # ls.optCuts[[s]] <- optCuts$par
}
cat("transform the train, valid, and test\n")
dt.pred.train <- as.data.table(sapply(ls.pred.train, print))
dt.pred.valid <- as.data.table(sapply(ls.pred.valid, print))
dt.pred.test <- as.data.table(sapply(ls.pred.test, print))
cat("transform the op\n")
dt.pred.valid.op <- as.data.table(sapply(ls.pred.valid.op, print))
dt.pred.test.op <- as.data.table(sapply(ls.pred.test.op, print))
# cat("transform optCuts\n")
# dt.optCuts <- as.data.table(sapply(ls.optCuts, print))

dt.pred.train
dt.pred.valid
dt.pred.test

dt.pred.valid.op
dt.pred.test.op

# dt.optCuts

cat("median combine the preds\n")
pred.train.final <- apply(dt.pred.valid, 1, function(x) median(x))
pred.valid.final <- apply(dt.pred.valid, 1, function(x) median(x))
pred.test.final <- apply(dt.pred.test, 1, function(x) median(x))

pred.valid.final.op <- apply(dt.pred.valid.op, 1, function(x) median(x))
pred.test.final.op <- apply(dt.pred.test.op, 1, function(x) median(x))

# cat("median combine the opCuts")
# opCuts.final <- apply(dt.optCuts, 1, function(x) median(x))

# cat("apply opCuts on pred.valid.final")
# cuts.valid.final <- c(min(pred.valid.final), opCuts.final, max(pred.valid.final))
# pred.valid.final.op <- as.integer(pred.valid.final, opCuts.final)

cat("check the score")
score <- ScoreQuadraticWeightedKappa(y.valid, round(pred.valid.final.op))
score
# 0.6601923 *
# 0.6592457 now is for raw, excluding impute 1
# 0.6597988 now is for raw, including imptue 1 and 2016
# 0.6608228 raw features with kmeans meta features
# 0.6608745 raw features with impute 1, without impute 2016
# 0.6603385 with square and cube Age, Wt, Ht, and BMI
# 0.6606217 with tsne and NewFeature1
# 0.6589659 raw with binary encode

################################
## 1.3 submit ##################
################################
submission = data.table(Id = dt.test$Id)
submission$Response = round(pred.test.final.op)
table(submission$Response)
# 1    2    3    4    5    6    7    8 
# 1715  934 1504 1693 2259 2672 3333 5655

# 1    2    3    4    5    6    7    8 
# 1568 1024 1536 1640 2178 2803 3127 5889 
write.csv(submission, "submit/011_xgb_poisson_recv_with_all_features.csv", row.names = FALSE) # 0.6601923 (highest) (LB 0.66819) *
write.csv(submission, "submit/013_xgb_poisson_recv_with_all_features_excl_impute_1.csv", row.names = FALSE) # 0.6601923 (LB 0.66719)
write.csv(submission, "submit/014_xgb_poisson_recv_with_raw_features_excl_impute_1.csv", row.names = FALSE) # 0.6592457 (LB 0.66677)
write.csv(submission, "submit/015_xgb_poisson_recv_with_raw_features_incl_impute_1_2016_with_kmeans_meta_features.csv", row.names = FALSE) # 0.6608228 (highest) (LB 0.66667)
write.csv(submission, "submit/016_xgb_poisson_recv_with_raw_features_incl_impute_1_.csv", row.names = FALSE) # 0.6608745 (highest) (LB 0.66809)
write.csv(submission, "submit/017_xgb_poisson_recv_with_square_cube_transform.csv", row.names = FALSE) # 0.6603385 (LB 0.66579)
write.csv(submission, "submit/018_xgb_poisson_recv_with_tsne_and_newfeature1.csv", row.names = FALSE) # 0.6603385 (LB 0.66579)
write.csv(submission, "submit/019_xgb_poisson_recv_with_binary_encode.csv", row.names = FALSE) # 0.6603385 (LB 0.66579)





