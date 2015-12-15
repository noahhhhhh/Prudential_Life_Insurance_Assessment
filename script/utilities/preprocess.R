############################################################################################
## 1. ColNAs ###############################################################################
############################################################################################
## Intro: summarise the stats of NAs of columns in a data table
## Args:
##  dt(data.table): a data table
##  method(character): "sum" - number of NAs; "mean" - NAs proportion of total
##  output(character): "all" - return all the output; "nonZero" - return only non-zero output
## Return(numeric): output of NA stats
ColNAs <- function(dt, method = "sum", output = "all"){
    out <- as.numeric()
    if (method == "sum"){
        out <- apply(dt, 2, function(x){sum(is.na(x))})
    } else if (method == "mean"){
        out <- apply(dt, 2, function(x){round(mean(is.na(x)), 2)})
    } else {
        return(F)
    }
    
    if (output == "all"){
        return(out)
    } else if (output == "nonZero"){
        return(out[out != 0])
    }
}

############################################################################################
## 2. ColUnique ############################################################################
############################################################################################
## Intro: summarise the number of unique values of columns in a data table
## Args:
##  dt(data.table): a data table
## Return(numeric): output of unique value stats
ColUnique <- function(dt){
    out <- apply(dt, 2, function(x){length(unique(x))})
    return(out)
}

############################################################################################
## 3. ConvertClass #########################################################################
############################################################################################
## Intro: convert class of columns in a data table
## Args:
##  dt(data.table): a data table
## Return(numeric): output of unique value stats
ColUnique <- function(dt){
    out <- apply(dt, 2, function(x){length(unique(x))})
    return(out)
}