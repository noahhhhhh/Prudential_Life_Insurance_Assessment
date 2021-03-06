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
# create a 3 folds
folds <- createFolds(dt.train$Response, k = 3, list = F)

# reproduce with m = 1 and n = 2
cat("initiating variables ...\n")

ls.pred.train <- list()
ls.pred.valid <- list()
ls.pred.test <- list()
ls.pred.train.op <- list()
ls.pred.valid.op <- list()
ls.pred.test.op <- list()
ls.pred.dist.train <- list()
ls.pred.dist.valid <- list()
ls.pred.dist.test <- list()
ls.optCuts <- list()

cat("training ...\n")
for(s in 1:15){
    # set up a score metric for folds
    pred.train <- rep(0, dim(dt.train)[1])
    pred.valid <- rep(0, dim(dt.valid)[1])
    pred.test <- rep(0, dim(dt.test)[1])
    
    pred.dist.train <- rep(0, dim(dt.train)[1])
    pred.dist.valid <- rep(0, dim(dt.valid)[1])
    pred.dist.test <- rep(0, dim(dt.test)[1])
    
    pred.train.ls <- list()
    pred.valid.ls <- list()
    pred.test.ls <- list()
    
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
                                , params = list(nthread = 8
                                                , eta = .025
                                                , min_child_weight = 20
                                                , max_depth = 8
                                                , subsample = .8
                                                , colsample_bytree = .8
                                                , metrics = "rmse"
                                )
                                , early.stop.round = 20
                                , maximize = F
                                , print.every.n = 150
                                , nrounds = 18000
                                , watchlist = list(valid = dmx.valid.fold, train = dmx.train.fold)
                                , verbose = T
        )
        # predict on train, valid, and test
        pred.train.temp <- predict(cv.xgb.out, dmx.train)
        pred.valid.temp <- predict(cv.xgb.out, dmx.valid)
        pred.test.temp <- predict(cv.xgb.out, x.test)
        
        pred.train <- pred.train + pred.train.temp
        pred.valid <- pred.valid + pred.valid.temp
        pred.test <- pred.test + pred.test.temp
        
        # predict on folded train and test set
        pred.train.fold <- predict(cv.xgb.out, dmx.train.fold)
        pred.valid.fold <- predict(cv.xgb.out, dmx.valid.fold)
        
        cat("optimising the cuts on pred.train.fold ...\n")
        SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
            cuts <- c(min(pred.train.fold), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.train.fold))
            pred <- as.integer(cut2(pred.train.fold, cuts))
            err <- ScoreQuadraticWeightedKappa(pred, y.train.fold, 1, 8)
            return(-err)
        }
        optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
        optCuts
        
        cat("applying optCuts on train.fold ...\n") #
        cuts.train.fold <- c(min(pred.train.fold), optCuts$par, max(pred.train.fold)) #
        pred.train.fold.op <- as.integer(cut2(pred.train.fold, cuts.train.fold)) #
        pred.dist.train.fold <- y.train.fold - pred.train.fold.op #
        
        cat("applying optCuts on valid.fold ...\n") #
        cuts.valid.fold <- c(min(pred.valid.fold), optCuts$par, max(pred.valid.fold)) #
        pred.valid.fold.op <- as.integer(cut2(pred.valid.fold, cuts.valid.fold)) #
        pred.dist.valid.fold <- y.valid.fold - pred.valid.fold.op #
        
        # add the prediction on pred.train.fold
        dt.pred.train.fold <- dt.train.fold[, pred := pred.dist.train.fold]
        x.pred.train.fold <- model.matrix(pred ~., dt.pred.train.fold[, !c("Id", "isTest", "Response"), with = F])[, -1]
        y.pred.train.fold <- dt.pred.train.fold$pred
        dmx.pred.train.fold <- xgb.DMatrix(data =  x.pred.train.fold, label = y.pred.train.fold)
        
        # add the prediction on pred.valid.fold
        dt.pred.valid.fold <- dt.valid.fold[, pred := pred.dist.valid.fold]
        x.pred.valid.fold <- model.matrix(pred ~., dt.pred.valid.fold[, !c("Id", "isTest", "Response"), with = F])[, -1]
        y.pred.valid.fold <- dt.pred.valid.fold$pred
        dmx.pred.valid.fold <- xgb.DMatrix(data =  x.pred.valid.fold, label = y.pred.valid.fold)
        
        cat("train on the residual of train fold ...\n")
        set.seed(m * 8 + n * 64 + k * 512 + s * 1024)
        cv.xgb.out.dist <- xgb.train(data = dmx.pred.train.fold
                                , booster = "gbtree"
                                , objective = "reg:linear"
                                , params = list(nthread = 8
                                                , eta = .025
                                                , min_child_weight = 20
                                                , max_depth = 8
                                                , subsample = .8
                                                , colsample_bytree = .8
                                                , metrics = "rmse"
                                )
                                , early.stop.round = 20
                                , maximize = F
                                , print.every.n = 150
                                , nrounds = 18000
                                , watchlist = list(valid = dmx.pred.valid.fold, train = dmx.pred.train.fold)
                                , verbose = T
        )
        cat("predict the dist on fold.final ...\n")
        pred.dist.train.fold.final <- predict(cv.xgb.out.dist, dmx.pred.train.fold)
        pred.dist.valid.fold.final <- predict(cv.xgb.out.dist, dmx.pred.valid.fold)
        
        pred.train.fold.final <- pred.dist.train.fold.final + pred.train.fold.op
        pred.valid.fold.final <- pred.dist.valid.fold.final + pred.valid.fold.op
        
        cat("optimising the cuts on pred.train.fold.final ...\n")
        SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
            cuts <- c(min(pred.train.fold.final), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.train.fold.final))
            pred <- as.integer(cut2(pred.train.fold.final, cuts))
            err <- ScoreQuadraticWeightedKappa(pred, y.train.fold, 1, 8)
            return(-err)
        }
        optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
        optCuts
        
        cat("applying optCuts on valid.fold.final ...\n") #
        cuts.valid.fold.final <- c(min(pred.valid.fold.final), optCuts$par, max(pred.valid.fold.final)) #
        pred.valid.fold.op.final <- as.integer(cut2(pred.valid.fold.final, cuts.valid.fold.final)) #
        print(paste("loop", s, "; k", k, ": valid fold score -", ScoreQuadraticWeightedKappa(y.valid.fold, pred.valid.fold.op.final)))
        
        cat("predict the dist on train, valid, and test ...\n")
        pred.dist.train.final <- predict(cv.xgb.out.dist, dmx.train)
        pred.dist.valid.final <- predict(cv.xgb.out.dist, dmx.valid)
        pred.dist.test.final <- predict(cv.xgb.out.dist, x.test)
        
        # pred.train
        dt.pred.train.temp <- dt.train[, pred := pred.dist.train.final]
        x.pred.train.temp <- model.matrix(pred ~., dt.train[, !c("Id", "isTest", "Response"), with = F])[, -1]
        y.pred.train.temp <- dt.pred.train.temp$pred
        dmx.pred.train.temp <- xgb.DMatrix(data =  x.pred.train.temp, label = y.pred.train.temp)
        
        # add the prediction on pred.train.fold
        dt.pred.valid.temp <- dt.valid[, pred := pred.dist.valid.final]
        x.pred.valid.temp <- model.matrix(pred ~., dt.valid[, !c("Id", "isTest", "Response"), with = F])[, -1]
        y.pred.valid.temp <- dt.pred.valid.temp$pred
        dmx.pred.valid.temp <- xgb.DMatrix(data =  x.pred.valid.temp, label = y.pred.valid.temp)
        
        cat("optimising the cuts on pred.train.temp ...\n")
        SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
            cuts <- c(min(pred.train.temp), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.train.temp))
            pred <- as.integer(cut2(pred.train.temp, cuts))
            err <- ScoreQuadraticWeightedKappa(pred, y.train, 1, 8)
            return(-err)
        }
        optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
        optCuts
        
        cat("applying optCuts on valid.temp ...\n") #
        cuts.valid.temp <- c(min(pred.valid.temp), optCuts$par, max(pred.valid.temp)) #
        pred.valid.temp <- as.integer(cut2(pred.valid.temp, cuts.valid.temp)) #
        print(paste("loop", s, "; k", k, ": valid score -", ScoreQuadraticWeightedKappa(y.valid, pred.valid.temp)))
        
        cat("applying optCuts on test.temp ...\n") #
        cuts.test.temp <- c(min(pred.test.temp), optCuts$par, max(pred.test.temp)) #
        pred.test.temp <- as.integer(cut2(pred.test.temp, cuts.test.temp)) #
        
        pred.train.final <- pred.dist.train.final + pred.train.temp
        pred.valid.final <- pred.dist.valid.final + pred.valid.temp
        pred.test.final <- pred.dist.test.final + pred.test.temp
        
        cat("optimising the cuts on pred.train.final ...\n")
        SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
            cuts <- c(min(pred.train.final), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.train.final))
            pred <- as.integer(cut2(pred.train.final, cuts))
            err <- ScoreQuadraticWeightedKappa(pred, y.train, 1, 8)
            return(-err)
        }
        optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
        optCuts
        
        cat("applying optCuts on valid.final ...\n") #
        cuts.valid.final <- c(min(pred.valid.final), optCuts$par, max(pred.valid.final)) #
        pred.valid.op.final <- as.integer(cut2(pred.valid.final, cuts.valid.final)) #
        print(paste("loop", s, "; k", k, ": valid final score -", ScoreQuadraticWeightedKappa(y.valid, pred.valid.op.final)))
        
        cat("applying optCuts on test.final ...\n") #
        cuts.test.final <- c(min(pred.test.final), optCuts$par, max(pred.test.final)) #
        pred.test.op.final <- as.integer(cut2(pred.test.final, cuts.test.final)) #
        
        pred.valid <- pred.valid + pred.valid.op.final
        pred.test <- pred.test + pred.test.op.final
        
        pred.valid.ls[[k]] <- pred.valid.op.final
        pred.test.ls[[k]] <- pred.test.op.final
    }
    
    pred.valid <- pred.valid / 3
    pred.test <- pred.test / 3
    
    dt.pred.valid.ls <- as.data.table(sapply(pred.valid.ls, print))
    dt.pred.test.ls <- as.data.table(sapply(pred.test.ls, print))
    
    pred.valid.op <- apply(dt.pred.valid.ls, 1, function(x) median(x))
    pred.test.op <- apply(dt.pred.test.ls, 1, function(x) median(x))
    
    print(paste("loop", s, ": valid score -", ScoreQuadraticWeightedKappa(y.valid, round(pred.valid.op))))
    
    cat("combining the optimised predictions ...\n")
    
    ls.pred.valid.op[[s]] <- pred.valid.op
    ls.pred.test.op[[s]] <- pred.test.op
    
    ls.optCuts[[s]] <- optCuts$par
}
cat("transform the op\n")
dt.pred.valid.op <- as.data.table(sapply(ls.pred.valid.op, print))
dt.pred.test.op <- as.data.table(sapply(ls.pred.test.op, print))

cat("transform optCuts\n")
dt.optCuts <- as.data.table(sapply(ls.optCuts, print))

cat("median combine the preds\n")
pred.valid.final.op <- apply(dt.pred.valid.op, 1, function(x) median(x))
pred.test.final.op <- apply(dt.pred.test.op, 1, function(x) median(x))

cat("median combine the opCuts")
opCuts.final <- apply(dt.optCuts, 1, function(x) median(x))

# cat("apply opCuts on pred.valid.final")
# cuts.valid.final <- c(min(pred.valid.final), opCuts.final, max(pred.valid.final))
# pred.valid.final.op <- as.integer(pred.valid.final, opCuts.final)

cat("check the score")
score.valid <- ScoreQuadraticWeightedKappa(y.valid, round(pred.valid.final.op))
score.valid
# 0.6601923 *
# 0.6592457 now is for raw, excluding impute 1
# 0.6597988 now is for raw, including imptue 1 and 2016
# 0.6608228 raw features with kmeans meta features
# 0.6608745 raw features with impute 1, without impute 2016
# 0.6603385 with square and cube Age, Wt, Ht, and BMI

################################
## 1.3 submit ##################
################################
submission = data.table(Id = dt.test$Id)
submission$Response = round(pred.test.final.op)
table(submission$Response)
# 1    2    3    4    5    6    7    8 
# 1715  934 1504 1693 2259 2672 3333 5655
write.csv(submission, "submit/011_xgb_poisson_recv_with_all_features.csv", row.names = FALSE) # 0.6601923 (highest) (LB 0.66819) *
write.csv(submission, "submit/013_xgb_poisson_recv_with_all_features_excl_impute_1.csv", row.names = FALSE) # 0.6601923 (LB 0.66719)
write.csv(submission, "submit/014_xgb_poisson_recv_with_raw_features_excl_impute_1.csv", row.names = FALSE) # 0.6592457 (LB 0.66677)
write.csv(submission, "submit/015_xgb_poisson_recv_with_raw_features_incl_impute_1_2016_with_kmeans_meta_features.csv", row.names = FALSE) # 0.6608228 (highest) (LB 0.66667)
write.csv(submission, "submit/016_xgb_poisson_recv_with_raw_features_incl_impute_1_.csv", row.names = FALSE) # 0.6608745 (highest) (LB 0.66809)
write.csv(submission, "submit/017_xgb_poisson_recv_with_square_cube_transform.csv", row.names = FALSE) # 0.6603385 (LB 0.66579)






