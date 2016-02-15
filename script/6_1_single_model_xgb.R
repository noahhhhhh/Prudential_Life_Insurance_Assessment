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
ind.train <- createDataPartition(dt.preprocessed.combine[isTest == 0]$Response, p = .8, list = F) # remember to change it to .66
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

evalerror <- function(preds, dtrain){
    labels <- getinfo(dtrain, "label")
    err <- ScoreQuadraticWeightedKappa(labels,round(preds))
    
    return(list(metric = "kappa", value = err))
}

cat("training ...\n")
for(s in 1:7){
    # set up a score metric for folds
    pred.train <- rep(0, dim(dt.train)[1])
    pred.valid <- rep(0, dim(dt.valid)[1])
    pred.test <- rep(0, dim(dt.test)[1])
    
    # for(k in 1:3){ # folds
    #         set.seed(m * 8 + n * 64 + k * 512 + s * 1024)
    #         # dmx.train.fold
    #         dt.train.fold <- dt.train[folds != k]
    #         x.train.fold <- model.matrix(Response ~., dt.train.fold[, !c("Id", "isTest"), with = F])[, -1]
    #         # x.train.fold <- data.matrix(dt.train.fold[, !c("Id", "isTest", "Response"), with = F])
    #         y.train.fold <- dt.train.fold$Response
    #         dmx.train.fold <- xgb.DMatrix(data =  x.train.fold, label = y.train.fold)
    #         # dmx.valid.fold
    #         dt.valid.fold <- dt.train[folds == k]
    #         x.valid.fold <- model.matrix(Response ~., dt.valid.fold[, !c("Id", "isTest"), with = F])[, -1]
    #         # x.valid.fold <- data.matrix(dt.valid.fold[, !c("Id", "isTest", "Response"), with = F])
    #         y.valid.fold <- dt.valid.fold$Response
    #         dmx.valid.fold <- xgb.DMatrix(data =  x.valid.fold, label = y.valid.fold)
    # train
    set.seed(m * 8 + n * 64 + s * 1024)
    cv.xgb.out <- xgb.train(data = dmx.train
                            , booster = "gbtree"
                            , objective = "count:poisson"
                            # , objective = "reg:linear"
                            , params = list(nthread = 8
                                            , eta = .025
                                            , min_child_weight = 100
                                            , max_depth = 8
                                            , subsample = .8
                                            , colsample_bytree = .8
                                            # , metrics = "rmse"
                            )
                            , feval = evalerror #
                            , early.stop.round = 100
                            # , maximize = F
                            , maximize = T #
                            , print.every.n = 150
                            , nrounds = 18000
                            , watchlist = list(valid = dmx.valid, train = dmx.train)
                            , verbose = T
    )
    pred.train <- predict(cv.xgb.out, dmx.train)
    pred.valid <- predict(cv.xgb.out, dmx.valid)
    pred.test <- predict(cv.xgb.out, x.test)
    # }
    
    #     pred.train <- pred.train / 3
    #     pred.valid <- pred.valid / 3
    #     pred.test <- pred.test / 3
    
    # optCutsPar <- rep(0, 7)
    # for (j in 1:6){
        set.seed(m * 8 + n * 64 + s * 1024)
        trainForOpt <- sample(length(pred.train), length(pred.train) * .8)
        pred.train.forOpt <- pred.train[trainForOpt]
        cat("optimising the cuts on pred.train ...\n")
        SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
            cuts <- c(min(pred.train.forOpt), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.train.forOpt))
            pred <- as.integer(cut2(pred.train.forOpt, cuts))
            err <- ScoreQuadraticWeightedKappa(pred, y.train[trainForOpt], 1, 8)
            return(-err)
        }
        optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
        # optCutsPar <- optCutsPar + optCuts$par
    # }
    # optCuts$par <- optCutsPar / 6
        # 0.656536928228719
    cat("applying optCuts on valid ...\n")
    cuts.valid <- c(min(pred.valid), optCuts$par, max(pred.valid))
    pred.valid.op <- as.integer(cut2(pred.valid, cuts.valid))
    print(paste("loop", s, ": valid score -", ScoreQuadraticWeightedKappa(y.valid, pred.valid.op)))
    # [1] "loop 1 : valid score - 0.662693668247028" (-1 as impute)
    # [1] "loop 1 : valid score - 0.664..." (-1 as impute plus all engineed features)
    
    cat("applying optCuts on test ...\n")
    cuts.test <- c(min(pred.test), optCuts$par, max(pred.test))
    pred.test.op <- as.integer(cut2(pred.test, cuts.test))
    
    cat("combining the optimised predictions ...\n")
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

# dt.optCuts

cat("median combine the preds\n")
pred.train.final <- apply(dt.pred.train, 1, function(x) mean(x))
pred.valid.final <- apply(dt.pred.valid, 1, function(x) mean(x))
pred.test.final <- apply(dt.pred.test, 1, function(x) mean(x))
# save
save(pred.train.final, pred.valid.final, pred.test.final, file = "model/xgb.RData")

pred.valid.final.op <- apply(dt.pred.valid.op, 1, function(x) median(x))
pred.test.final.op <- apply(dt.pred.test.op, 1, function(x) median(x))

cat("median combine the opCuts")
opCuts.final <- apply(dt.optCuts, 1, function(x) mean(x))

# cat("apply opCuts on pred.valid.final")
# cuts.valid.final <- c(min(pred.valid.final), opCuts.final, max(pred.valid.final))
# pred.valid.final.op <- as.integer(pred.valid.final, opCuts.final)

cat("check the score")
score <- ScoreQuadraticWeightedKappa(y.valid, round(pred.valid.final.op))
score
# 0.6601923
# 0.6592457 now is for raw, excluding impute 1
# 0.6597988 now is for raw, including imptue 1 and 2016
# 0.6608228 raw features with kmeans meta features
# 0.6608745 raw features with impute 1, without impute 2016
# 0.6603385 with square and cube Age, Wt, Ht, and BMI
# 0.6606217 with tsne and NewFeature1
# 0.6589659 raw with binary encode
# 0.6640989 with -1 as the impute and all engineed features (lb 0.66857)
# 0.6633673 same as above but 80% of training set used to train optCuts (lb 0.66944)
# 0.6645372 same as above but with dummy vars (lb 0.66953)
# 0.6578306 same as above but with 80% train and 20% valid (lb 0.67114)
# 0.6576221 same as above but with product_2_num (lb 0.67109)
# 0.6576881 sane as above but with product_2_1 and without group features (lb 0.67045)
# 0.6565563 same as above but with group features (all engineed features) and 100 min_child_weight (lb 0.67343)
# 0.6574313 same as above but with tsne (lb 0.67426) *
# 0.6569404 same as above but optim cut after all, rather than in each loop (lb 0.67131)
# 0.6543513 same as above, just train fold (k = 10) (lb 0.66891)
# 0.6636679 same as above, just train on 90% train and 6 fold prediciton (lb 0.66994)
# 0.6567266 tsnes and kmeans (lb 0.67024)
# 0.6573504 with all_kmeans (lb 0.67077)
# 0.6561871 with lr faetures (lb 0.67183)
# 0.6591701 using lr's ind.train with tsne and group features (lb 0.66875)
# 0.657739 try to reproduce with rmse (lb 0.67258)
# 0.6565868 try to reproduce with eval (lb 0.67286)

################################
## 1.3 submit ##################
################################
submission = data.table(Id = dt.test$Id)
submission$Response = round(pred.test.final.op)
table(submission$Response)

write.csv(submission, "submit/011_xgb_poisson_recv_with_all_features.csv", row.names = FALSE) # 0.6601923 (highest) (LB 0.66819)
write.csv(submission, "submit/013_xgb_poisson_recv_with_all_features_excl_impute_1.csv", row.names = FALSE) # 0.6601923 (LB 0.66719)
write.csv(submission, "submit/014_xgb_poisson_recv_with_raw_features_excl_impute_1.csv", row.names = FALSE) # 0.6592457 (LB 0.66677)
write.csv(submission, "submit/015_xgb_poisson_recv_with_raw_features_incl_impute_1_2016_with_kmeans_meta_features.csv", row.names = FALSE) # 0.6608228 (highest) (LB 0.66667)
write.csv(submission, "submit/016_xgb_poisson_recv_with_raw_features_incl_impute_1_.csv", row.names = FALSE) # 0.6608745 (highest) (LB 0.66809)
write.csv(submission, "submit/017_xgb_poisson_recv_with_square_cube_transform.csv", row.names = FALSE) # 0.6603385 (LB 0.66579)
write.csv(submission, "submit/018_xgb_poisson_recv_with_tsne_and_newfeature1.csv", row.names = FALSE) # 0.6603385 (LB 0.66579)
write.csv(submission, "submit/019_xgb_poisson_recv_with_binary_encode.csv", row.names = FALSE) # 0.6603385 (LB 0.66579)
write.csv(submission, "submit/020_xgb_poisson_recv_with_impute_1_and_all_engineed_features.csv", row.names = FALSE) # 0.6640989 (LB 0.66857)
write.csv(submission, "submit/021_xgb_poisson_benchmark_para_cv_with_impute_1_and_all_engineed_features.csv", row.names = FALSE) # 0.6640989 (LB 0.66857)
write.csv(submission, "submit/022_xgb_poisson_benchmark_para_cv_with_impute_1_and_all_engineed_features_with_08percent_optcuts.csv", row.names = FALSE) # 0.6645372 (LB 0.66944)
write.csv(submission, "submit/023_xgb_poisson_benchmark_para_cv_with_impute_1_and_all_engineed_features_with_dummy_vars_with_08percent_optcuts.csv", row.names = FALSE) # 0.6645372 (LB 0.66953)
write.csv(submission, "submit/024_xgb_poisson_recv_feval_08trai02valid_with_impute_1_and_all_engineed_features_with_dummy_vars_with_08percent_optcuts.csv", row.names = FALSE) # 0.6578306 (LB 0.67114)
write.csv(submission, "submit/025_xgb_poisson_recv_feval_08trai02valid_with_impute_1_and_all_engineed_features_with_dummy_vars_with_08percent_optcuts_with_product_2_num.csv", row.names = FALSE) # 0.6578306 (LB 0.67114)
write.csv(submission, "submit/026_xgb_poisson_recv_feval_08trai02valid_with_impute_1_and_all_engineed_features_with_dummy_vars_with_08percent_optcuts_with_product_2_num_and_product_2_1_without_group_features.csv", row.names = FALSE) # 0.6576881 (LB 0.67045)
write.csv(submission, "submit/027_xgb_poisson_recv_feval_08trai02valid_with_impute_1_and_all_engineed_features_with_dummy_vars_with_08percent_optcuts_with_product_2_num_and_product_2_1_without_group_features_min_child_weight_100.csv", row.names = FALSE) # 0.6576881 (LB 0.67045)
write.csv(submission, "submit/028_xgb_poisson_recv_feval_08trai02valid_with_impute_1_and_all_engineed_features_with_dummy_vars_with_08percent_optcuts_with_product_2_num_and_product_2_1_without_group_features_min_child_weight_100_tsne.csv", row.names = FALSE) # 0.6574313 (LB 0.67426) *
write.csv(submission, "submit/029_xgb_poisson_recv_feval_08trai02valid_with_impute_1_and_all_engineed_features_with_dummy_vars_with_08percent_optcuts_with_product_2_num_and_product_2_1_without_group_features_min_child_weight_100_tsne_optim_afterall.csv", row.names = FALSE) # 0.6569404 (LB 0.67131)
write.csv(submission, "submit/030_xgb_poisson_recv_feval_08trai02valid_with_impute_1_and_all_engineed_features_with_dummy_vars_with_08percent_optcuts_with_product_2_num_and_product_2_1_without_group_features_min_child_weight_100_tsne_optim_afterall_trainfold.csv", row.names = FALSE) # 0.6543513 (LB 0.66891)
write.csv(submission, "submit/031_xgb_poisson_recv_feval_09trai01valid_with_impute_1_and_all_engineed_features_with_dummy_vars_with_08percent_optcuts_with_product_2_num_and_product_2_1_without_group_features_min_child_weight_100_tsne_10k.csv", row.names = FALSE) # 0.6543513 (LB 0.66891)
write.csv(submission, "submit/032_xgb_poisson_recv_feval_09trai01valid_with_impute_1_and_all_engineed_features_with_dummy_vars_with_08percent_optcuts_with_product_2_num_and_product_2_1_without_group_features_min_child_weight_100_tnses_kmeans.csv", row.names = FALSE) # 0.6543513 (LB 0.66891)
write.csv(submission, "submit/033_xgb_poisson_recv_feval_09trai01valid_with_impute_1_and_all_engineed_features_with_dummy_vars_with_08percent_optcuts_with_product_2_num_and_product_2_1_without_group_features_min_child_weight_100_tnse_and_all_kmeans.csv", row.names = FALSE) # 0.6573504 (LB 0.67175)
write.csv(submission, "submit/034_xgb_with_Family_Hist_Kmeans_Medical_Keyword_Kmeans_eta_002.csv", row.names = FALSE) # 0.6573504 (LB 0.67175)
write.csv(submission, "submit/035_xgb_with_regression_features.csv", row.names = FALSE) # 0.6561871 (LB 0.67183)
write.csv(submission, "submit/036_xgb_using_lrs_ind_train.csv", row.names = FALSE) # 0.6561871 (LB 0.67183)
write.csv(submission, "submit/037_xgb_try_to_reproduce.csv", row.names = FALSE) # 0.657739 (LB 0.67258)
write.csv(submission, "submit/038_xgb_try_to_reproduce.csv", row.names = FALSE) # 0.6565868 (LB 0.67286)





