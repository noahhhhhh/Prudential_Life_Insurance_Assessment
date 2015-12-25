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
    err <- ScoreQuadraticWeightedKappa(labels,round(preds))
    
    return(list(metric = "QW Kappa", value = err))
}