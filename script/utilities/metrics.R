############################################################################################
## 1. QuadraticWeightedKappa ###############################################################
############################################################################################
## Intro: QuadraticWeightedKappa
## Args:
##  preds(numeric/integer vector): prediction vector
##  dtrain(xgb.DMatrix): a training xgb.DMatrix
## Return(list): output of QuadraticWeightedKappa
QuadraticWeightedKappa <- function(preds, dtrain){
    require(Metrics)
    labels <- getinfo(dtrain, "label")
    x <- seq(1.5, 7.5, by = 1) # added
    cuts <- c(-Inf, x[1], x[2], x[3], x[4], x[5], x[6], x[7], Inf) # added
    preds = as.integer(Hmisc::cut2(preds, cuts)) # added
    score <- ScoreQuadraticWeightedKappa(preds, labels, 1, 8) # added
    # score <- ScoreQuadraticWeightedKappa(labels,round(preds))
    
    return(list(metric = "kappa", value = -score))
}