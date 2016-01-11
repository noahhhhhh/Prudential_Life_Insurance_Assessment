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

############################################################################################
## 4. ConvertNonNumFactorToNumFactor #######################################################
############################################################################################
## Intro: Convert non-numeric factor to a numeric factor
## Args:
##  dt(data.table): a data table
##  col(a vector of characters): names of a targeted column
## Return(data.table): output of a data table with the additional binary columns
ConvertNonNumFactorToNumFactor <- function(dt, col){
    # unique values dict
    dict.uniq <- data.table(ID = rownames(unique(dt[, col, with = F]))
                            , unique(dt[, col, with = F]))
    # ID corresponding to col
    dt.col <- merge(dt, dict.uniq, by = col, all.x = T)
    # set the name of the new col
    colname <- paste(col, "_toNum", sep = "")
    setnames(dt.col, names(dt.col), c(names(dt.col)[-length(names(dt.col))], colname))
    
    return(dt.col)
}

############################################################################################
## 5. Noise ################################################################################
############################################################################################
## Intro: add noise into a data table. This function references Ivan Liu.
## Args:
##  dt(data.table): a data table
##  noise_l(numeric): lower noise
##  noise_u(numeric): upper noise
##  col_excl(a vector of characters): names of columns not included which do not apply the noise
## Return(data.table): output of a data table after adding noise
Noise <- function(dt, noise_l = -.00001, noise_u = .00001, col_excl){
    dim(dt)
    dt_noise <- apply(dt[, !col_excl, with = F], 2, function(x){
        runif(n = length(x), min = noise_l * diff(range(x)), max = noise_u * diff(range(x)))
    })
    dt_noise <- dt[, !col_excl, with = F] + dt_noise
    
    dt_noise <- data.table(dt_noise, dt[, col_excl, with = F])
    dt <- rbind(dt, dt_noise)
    dim(dt)
    dt <- dt[sample(1:nrow(dt), size = nrow(dt))]
    return(dt)
}














