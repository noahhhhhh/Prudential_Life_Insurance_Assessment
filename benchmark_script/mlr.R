# Created by Giuseppe Casalicchio
rm(list = ls()); gc();
setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Prudential_Life_Insurance_Assessment/")
load("data/data_preprocess/dt_proprocess_combine.RData")
library(Metrics)
library(Hmisc)
library(xgboost)
library(checkmate)
library(mlr) 
require(data.table)
require(caret)
require(Metrics)
# Tutorial: https://mlr-org.github.io/mlr-tutorial/release/html/
# We are on Github, feel free to contribute or star us: https://github.com/mlr-org/mlr
##############################################
# Disregard the code until line 75, its because kaggle has an old mlr package version installed
makeRLearner.regr.xgboost = function() {
    makeRLearnerRegr(
        cl = "regr.xgboost",
        package = "xgboost",
        par.set = makeParamSet(
            # we pass all of what goes in 'params' directly to ... of xgboost
            #makeUntypedLearnerParam(id = "params", default = list()),
            makeDiscreteLearnerParam(id = "booster", default = "gbtree", values = c("gbtree", "gblinear")),
            makeIntegerLearnerParam(id = "silent", default = 0),
            makeNumericLearnerParam(id = "eta", default = 0.3, lower = 0),
            makeNumericLearnerParam(id = "gamma", default = 0, lower = 0),
            makeIntegerLearnerParam(id = "max_depth", default = 6, lower = 0),
            makeNumericLearnerParam(id = "min_child_weight", default = 1, lower = 0),
            makeNumericLearnerParam(id = "subsample", default = 1, lower = 0, upper = 1),
            makeNumericLearnerParam(id = "colsample_bytree", default = 1, lower = 0, upper = 1),
            makeIntegerLearnerParam(id = "num_parallel_tree", default = 1, lower = 1),
            makeNumericLearnerParam(id = "lambda", default = 0, lower = 0),
            makeNumericLearnerParam(id = "lambda_bias", default = 0, lower = 0),
            makeNumericLearnerParam(id = "alpha", default = 0, lower = 0),
            makeUntypedLearnerParam(id = "objective", default = "reg:linear"),
            makeUntypedLearnerParam(id = "eval_metric", default = "rmse"),
            makeNumericLearnerParam(id = "base_score", default = 0.5),
            
            makeNumericLearnerParam(id = "missing", default = 0),
            makeIntegerLearnerParam(id = "nthread", default = 16, lower = 1),
            makeIntegerLearnerParam(id = "nrounds", default = 1, lower = 1),
            makeUntypedLearnerParam(id = "feval", default = NULL),
            makeIntegerLearnerParam(id = "verbose", default = 1, lower = 0, upper = 2),
            makeIntegerLearnerParam(id = "print.every.n", default = 1, lower = 1),
            makeIntegerLearnerParam(id = "early.stop.round", default = 1, lower = 1),
            makeLogicalLearnerParam(id = "maximize", default = FALSE)
        ),
        par.vals = list(nrounds = 1),
        properties = c("numerics", "factors", "weights"),
        name = "eXtreme Gradient Boosting",
        short.name = "xgboost",
        note = "All settings are passed directly, rather than through `xgboost`'s `params` argument. `nrounds` has been set to `1` by default."
    )
}

trainLearner.regr.xgboost = function(.learner, .task, .subset, .weights = NULL,  ...) {
    td = getTaskDescription(.task)
    data = getTaskData(.task, .subset, target.extra = TRUE)
    target = data$target
    data = data.matrix(data$data)
    
    parlist = list(...)
    obj = parlist$objective
    if (testNull(obj)) {
        obj = "reg:linear"
    }
    
    if (testNull(.weights)) {
        xgboost::xgboost(data = data, label = target, objective = obj, ...)
    } else {
        xgb.dmat = xgboost::xgb.DMatrix(data = data, label = target, weight = .weights)
        xgboost::xgboost(data = xgb.dmat, label = NULL, objective = obj, ...)
    }
}

predictLearner.regr.xgboost = function(.learner, .model, .newdata, ...) {
    td = .model$task.desc
    m = .model$learner.model
    xgboost::predict(m, newdata = data.matrix(.newdata), ...)
}
#####################################

## Read Data
# train = read.csv("data/data_raw/train.csv", header = TRUE)
# test = read.csv("data/data_raw/test.csv", header = TRUE)
# test$Response = 0

train <- dt.preprocessed.combine[isTest == 0, !c("isTest"), with = F]
test <- dt.preprocessed.combine[isTest == 1, !c("isTest"), with = F]

## store Id column and remove it from the train and test data
testId = test$Id
# train$Id = test$Id = NULL
train[, Id := NULL]
test[, Id := NULL]

## create mlr task and convert factors to dummy features
trainTask = makeRegrTask(data = train, target = "Response")
trainTask = createDummyFeatures(trainTask)
testTask = makeRegrTask(data = test, target = "Response")
testTask = createDummyFeatures(testTask)

## create mlr learner
set.seed(1)
lrn = makeLearner("regr.xgboost")
lrn$par.vals = list(
    #nthread             = 30,
    nrounds             = 100,
    print.every.n       = 50,
    objective           = "reg:linear"
)
# missing values will be imputed by their median
lrn = makeImputeWrapper(lrn, classes = list(numeric = imputeMedian(), integer = imputeMedian()))

## Create Evaluation Function
SQWKfun = function(x = seq(1.5, 7.5, by = 1), pred) {
    preds = pred$data$response
    true = pred$data$truth 
    cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
    preds = as.numeric(Hmisc::cut2(preds, cuts))
    err = Metrics::ScoreQuadraticWeightedKappa(preds, true, 1, 8)
    return(-err)
}
SQWK = makeMeasure(id = "SQWK", minimize = FALSE, properties = c("regr"), best = 1, worst = 0,
                   fun = function(task, model, pred, feats, extra.args) {
                       return(-SQWKfun(x = seq(1.5, 7.5, by = 1), pred))
                   })

## This is how you could do hyperparameter tuning
# # 1) Define the set of parameters you want to tune (here 'eta')
# ps = makeParamSet(
#   makeNumericParam("eta", lower = 0.1, upper = 0.3)
# )
# # 2) Use 3-fold Cross-Validation to measure improvements
# rdesc = makeResampleDesc("CV", iters = 3L)
# # 3) Here we use Random Search (with 10 Iterations) to find the optimal hyperparameter
# ctrl =  makeTuneControlRandom(budget = 10, maxit = 10)
# # 4) now use the learner on the training Task with the 3-fold CV to optimize your set of parameters and evaluate it with SQWK
# res = tuneParams(lrn, task = trainTask, resampling = rdesc, par.set = ps, control = ctrl, measures = SQWK)
# res
# # 5) set the optimal hyperparameter
# lrn = setHyperPars(lrn, par.vals = res$x)

## now try to find the optimal cutpoints that maximises the SQWK measure based on the Cross-Validated predictions
cv = crossval(lrn, trainTask, iter = 3, measures = SQWK, show.info = TRUE)
# [Resample] cross-validation iter: 1
# [0]	train-rmse:4.237062
# [50]	train-rmse:1.638368
# [Resample] cross-validation iter: 2
# [0]	train-rmse:4.226523
# [50]	train-rmse:1.646894
# [Resample] cross-validation iter: 3
# [0]	train-rmse:4.241641
# [50]	train-rmse:1.650577
# [Resample] Result: SQWK.test.mean= 0.6
optCuts = optim(seq(1.5, 7.5, by = 1), SQWKfun, pred = cv$pred)
optCuts
# $par
# [1] 1.571734 3.414594 4.144019 4.888056 5.537057 6.251994 6.834168
## now train the model on all training data
tr = train(lrn, trainTask)
# 002_mlr_xgb_sample_code
# [0]	train-rmse:4.235723
# [50]	train-rmse:1.695176
## predict using the optimal cut-points 
pred = predict(tr, testTask)
preds = as.numeric(Hmisc::cut2(pred$data$response, c(-Inf, optCuts$par, Inf)))
table(preds)

## create submission file
submission = data.frame(Id = testId)
submission$Response = as.integer(preds)
write.csv(submission, "submit/002_mlr_xgb_sample_code_nzvV1.R", row.names = FALSE)


