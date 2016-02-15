#blend
rm(list = ls()); gc();
load("data/data_preprocess/dt_proprocess_combine.RData")
load("model/lr.RData")
cat("init train, valid, test")
head(ind.train)
dt.train <- dt.preprocessed.combine[isTest == 0][ind.train, ]
dt.valid <- dt.preprocessed.combine[isTest == 0][-ind.train, ]
dt.test <- dt.preprocessed.combine[isTest == 1]
dim(dt.train); dim(dt.valid); dim(dt.test)

cat("load the predictions")
load("model/xgb.RData")
pred.train.xgb <- pred.train.final
pred.valid.xgb <- pred.valid.final
pred.test.xgb <- pred.test.final

load("model/rf.RData")
pred.train.rf <- pred.train
pred.valid.rf <- pred.valid
pred.test.rf <- pred.test

load("model/lr.RData")
pred.train.lr <- response.train
pred.valid.lr <- response.valid
pred.test.lr <- response.test
ScoreQuadraticWeightedKappa(dt.valid$Response, pred.valid.lr)
# 0.6371514

cat("blending ...\n")
pred.train <- (pred.train.xgb * .7 + pred.train.rf * .3)
pred.valid <- (pred.valid.xgb * .7 + pred.valid.rf * .3)
pred.test <- (pred.test.xgb * .7 + pred.test.rf * .3)

cat("optimising the cuts on pred.train ...\n")
# pred.train <- pred.train.xgb #
SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
    trainForOpt <- sample(length(pred.train), length(pred.train) * .8)
    pred.train.forOpt <- pred.train[trainForOpt]
    cuts <- c(min(pred.train.forOpt), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.train.forOpt))
    pred <- as.integer(cut2(pred.train.forOpt, cuts))
    err <- ScoreQuadraticWeightedKappa(pred, dt.train$Response[trainForOpt], 1, 8)
    return(-err)
}
set.seed(2048)
optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
optCuts$par

# pred.valid <- pred.valid.xgb #
cat("applying optCuts on valid ...\n")
cuts.valid <- c(min(pred.valid), optCuts$par, max(pred.valid))
pred.valid.op <- as.integer(cut2(pred.valid, cuts.valid))
ScoreQuadraticWeightedKappa(dt.valid$Response, pred.valid.op)
# 0.6562745

pred.valid.op.op <- round(pred.valid.op * .9 + pred.valid.lr * .1)
pred.test.op.op <- round(pred.test * .9 + pred.test.lr * .1)

ScoreQuadraticWeightedKappa(dt.valid$Response, pred.valid.op.op)
# xgb: 0.6571425
# rf: 0.6334191
# lr: 0.6371514
# 0.6514746 combine

################################
## 1.3 submit ##################
################################
submission = data.table(Id = dt.test$Id)
submission$Response = round(pred.test.op.op)
table(submission$Response)
write.csv(submission, "submit/039_blend.csv", row.names = FALSE) # 0.6561871 (LB 0.67183)

