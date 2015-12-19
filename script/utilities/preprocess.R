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
    } else {
        return(F)
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
## 3. BinaryEncode #########################################################################
############################################################################################
## Intro: Binary Encoding for categorial features
## Args:
##  dt(data.table): a data table
##  cols(a vector of characters): names of targeted columns
## Return(data.table): output of a data table with the additional binary columns
BinaryEncode <- function(dt, cols){
    require(miscFuncs)
    require(stringr)
    for(col in cols){
        # unique values dict
        dict.uniq <- data.table(ID = rownames(unique(dt[, col, with = F]))
                                , unique(dt[, col, with = F]))
        # decimal to binary vector, e.g. 19 --> 10011
        vec.dec <- bin(dim(dict.uniq)[1])
        # length of the binary vector, e.g. 19 --> 10011 --> 5
        num.len <- length(vec.dec)
        # binary encoding
        # ID corresponding to col
        dt.col <- merge(dt[, col, with = F], dict.uniq, by = col, all.x = T)
        dt.col[, ID := as.integer(ID)]
        # encoded vector
        vec.bin <- unlist(apply(dt.col[, "ID", with = F], 1, function(x)(str_pad(paste(bin(x), collapse = ""), num.len, side = "left", pad = "0"))))
        # set up the col names and binary values
        vec.col  <- as.character()
        for (i in 1:num.len){
            vec.col[i] <- paste(col, "_bin_", i, sep = "")
            dt[, vec.col[i]:= as.integer(substr(vec.bin, i, i))]
        }
        dt <- dt[, !col, with = F]
    }
    return(dt)
}


















