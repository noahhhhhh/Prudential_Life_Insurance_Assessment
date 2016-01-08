rm(list = ls()); gc();
setwd("/Volumes/Data Science/Google Drive/data_science_competition/kaggle/Prudential_Life_Insurance_Assessment/")
load("data/data_preprocess/dt_proprocess_combine.RData")
source("script/utilities/metrics.R")
require(data.table)
require(caret)
require(Metrics)
require(Hmisc)
############################################################################################
## 1.0 linear regression ###################################################################
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
require(glmnet)
grid <- 10^seq(1, -5, length = 100)

vec.alpha <- as.numeric()
vec.lambda <- as.numeric()
vec.score <- as.numeric()
for (alpha in seq(0, 1, by = .1)){
    cat("train ...\n")
    md.lasso <- glmnet(x.train, y.train, alpha = alpha, lambda = grid, standardize = F, family = "gaussian")
    plot(md.lasso)
    
    cat("cv to choose λ ...\n")
    set.seed(1)
    cv.out <- cv.glmnet(x.train, y.train, alpha = alpha, lambda = grid, type.measure = "mse", family = "gaussian")
    plot(cv.out)
    cv.out$lambda
    bestlam <- cv.out$lambda.min
    
    cat("optimising the cuts on pred.train ...\n")
    pred.train <- predict(md.lasso , s = bestlam , newx = x.train)
    SQWKfun <- function(x = seq(1.5, 7.5, by = 1)){
        cuts <- c(min(pred.train), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(pred.train))
        pred <- as.integer(cut2(pred.train, cuts))
        err <- ScoreQuadraticWeightedKappa(pred, y.train, 1, 8)
        return(-err)
    }
    
    optCuts <- optim(seq(1.5, 7.5, by = 1), SQWKfun)
    optCuts
    
    cat("applying optCuts on valid ...\n")
    pred.valid <- predict(md.lasso , s = bestlam , newx = x.valid)
    cuts.valid <- c(min(pred.valid), optCuts$par, max(pred.valid))
    pred.valid.op <- as.integer(cut2(pred.valid, cuts.valid))
    score <- ScoreQuadraticWeightedKappa(y.valid, pred.valid.op)
    score
    # [1] 0.6211766 (alpha = .5, with 4 groups and raw continuous features)
    # [1] 0.5672361 (alpha = .5, with 4 groups and raw continuous features, removed nzv)
    # [1] 0.6080889 (alpha = .5, with 4 groups, no raw continuous features)
    # [1] 0.6134094 (alpha = .5, with raw features)
    # [1] 0.6211766 (alpha = .5, with 4 groups and raw continuous features, mulitinomial)
    
    print(paste("-------- alpha:", alpha, "; lambda:", bestlam, "; score:", score))
    
    vec.alpha <- c(vec.alpha, alpha)
    vec.lambda <- c(vec.lambda, bestlam)
    vec.score <- c(vec.score, score)
    
}

dt.result <- data.table(alpha = vec.alpha, lambda = vec.lambda, score = vec.score)
dt.result
# 1:   0.0 1.149757e-05 0.6212898
# 2:   0.1 1.149757e-05 0.6208407
# 3:   0.2 1.321941e-05 0.6202872
# 4:   0.3 1.149757e-05 0.6218982
# 5:   0.4 1.149757e-05 0.6209894
# 6:   0.5 1.149757e-05 0.6220926
# 7:   0.6 1.149757e-05 0.6191370
# 8:   0.7 1.149757e-05 0.6211452
# 9:   0.8 1.149757e-05 0.6207254
# 10:   0.9 1.149757e-05 0.6192319
# 11:   1.0 1.149757e-05 0.6225585 *
