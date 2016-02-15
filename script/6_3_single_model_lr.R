#Load libraries
library(MASS)
set.seed(25) #set seed (just in case)

#Function that imputes numeric missing values by column-by-column median substitution
#despite the for loops this is probably the best way to do it:
#http://stackoverflow.com/questions/23242389/median-imputation-using-sapply
manage_na <- function(datafra)
{
    for(i in 1:ncol(datafra))
    {
        if(is.numeric(datafra[,i]))
        {
            datafra[is.na(datafra[,i]),i] <- median(datafra[!is.na(datafra[,i]),i])
        }
    }
    datafra
}

#Nonlinear transformation of the labels (this is key to a decent score with a simple model)
#The hardcoded values were obtained by optimizing a CV score using simulated annealing
nonlintra <- function(y)
{
    hardcoded_values <- c(-1.6, 0.7, 0.3, 3.15, 4.53, 6.5, 6.77, 9.0)
    return(hardcoded_values[y])
}

predict_labels <- function(tra, test)
{
    ##----Reducing the number of features via ridge regression ---------------------------------------
    #Making a new dataframe with no Id, no Response, and only numeric variables, then using ridge regression
    #to remove some variables. The cutoff was optimized via cross-validation.
    tra_clean <- manage_na(tra[,-c(1,3,128)])
    ogg <- lm.ridge(tra$Response ~ ., data=tra_clean, lambda=0.5)
    impo <- tra_clean[,(abs(ogg$coef) > quantile(abs(ogg$coef), 0.382))] #only "important" variables left
    var_names <- names(impo) #their names
    ##------------------------------------------------------------------------------------------------
    
    ##----Nonlinear transformation of the labels -----------------------------------------------------
    y <- nonlintra(tra$Response) #nonlinear transformation of the labels
    ##------------------------------------------------------------------------------------------------
    
    ##----Feature engineering-------------------------------------------------------------------------
    #Quantile cut-off used to define custom variables 10 and 12 (step functions over BMI and BMI*Ins_Age)
    qbmic <- 0.8
    qbmic2 <- 0.9
    #Hand engineered features. Found by EDA (especially added variable plots), some parameters optimized
    #using cross validation. Nonlinear dependence on BMI and its interaction with age make intuitive sense.
    custom_var_1 <- as.numeric(tra$Medical_History_15 < 10.0)
    custom_var_1[is.na(custom_var_1)] <- 0.0 #impute these NAs with 0s, note that they were not median-imputed
    custom_var_3 <- as.numeric(tra$Product_Info_4 < 0.075)
    custom_var_4 <- as.numeric(tra$Product_Info_4 == 1)
    custom_var_6 <- (tra$BMI + 1.0)**2.0
    custom_var_7 <- (tra$BMI)**0.8
    custom_var_8 <- tra$Ins_Age**8.5
    custom_var_9 <- (tra$BMI*tra$Ins_Age)**2.5
    BMI_cutoff <- quantile(tra$BMI, qbmic)
    custom_var_10 <- as.numeric(tra$BMI > BMI_cutoff)
    custom_var_11 <- (tra$BMI*tra$Product_Info_4)**0.9
    ageBMI_cutoff <- quantile(tra$Ins_Age*tra$BMI, qbmic2)
    custom_var_12 <- as.numeric(tra$Ins_Age*tra$BMI > ageBMI_cutoff)
    custom_var_13 <- (tra$BMI*tra$Medical_Keyword_3 + 0.5)**3.0
    #Add the custom variables to the important variable dataframe
    impo <- cbind(impo, custom_var_1, custom_var_3, custom_var_4, custom_var_6, custom_var_7, custom_var_8, custom_var_9, custom_var_10, custom_var_11, custom_var_12, custom_var_13)
    #Remove weight and height, they are very correlated with BMI (we know, but VIFs can be used to show this)
    impo <- impo[,!(names(impo) %in% c("Ht", "Wt"))]
    #Same features as above
    custom_var_1 <- as.numeric(test$Medical_History_15 < 10.0)
    custom_var_3 <- as.numeric(test$Product_Info_4 < 0.075)
    custom_var_4 <- as.numeric(test$Product_Info_4 == 1)
    custom_var_1[is.na(custom_var_1)] <- 0.0
    custom_var_6 <- (test$BMI + 1.0)**2.0
    custom_var_7 <- (test$BMI)**0.8
    custom_var_8 <- test$Ins_Age**8.5
    custom_var_9 <- (test$BMI*test$Ins_Age)**2.5
    custom_var_10 <- as.numeric(test$BMI > BMI_cutoff)
    custom_var_11 <- (test$BMI*test$Product_Info_4)**0.9
    custom_var_12 <- as.numeric(test$Ins_Age*test$BMI > ageBMI_cutoff)	
    custom_var_13 <- (test$BMI*test$Medical_Keyword_3 + 0.5)**3.0
    #Make important variable dataframe for test as well
    tempo <- manage_na(test[,var_names])
    tempo <- cbind(tempo, custom_var_1, custom_var_3, custom_var_4, custom_var_6, custom_var_7, custom_var_8, custom_var_9, custom_var_10, custom_var_11, custom_var_12, custom_var_13)
    #Remove height and weight from there too
    tempo <- tempo[,!(names(tempo) %in% c("Ht", "Wt"))]
    ##------------------------------------------------------------------------------------------------
    
    ##----Fitting the linear model -------------------------------------------------------------------
    fa <- tra[,3] #put the factor back in		
    linear_model <- lm(y ~ ., data=cbind(impo,fa)) #fit the model
    ##------------------------------------------------------------------------------------------------
    
    ##----Predict the response using the linear model ------------------------------------------------
    fa <- test[,3] 
    Response <- predict(linear_model, newdata = cbind(tempo,fa))
    ##------------------------------------------------------------------------------------------------
    
    ##----Round using custom cutoffs. These were also optimized with CV ------------------------------
    hardcoded_cutoffs <- c(0.8717, 0.9034, 0.8119, 0.7567, 0.6588, 0.2360, 0.0490)
    
    Response[Response < 1] <- 1
    Response[Response > (7 + hardcoded_cutoffs[7])] <- 8
    for(i in 1:6)
    {
        lowcut <- hardcoded_cutoffs[i]
        hicut <- hardcoded_cutoffs[(i + 1)]
        condi <- (Response > (i + lowcut)) & (Response < (i + 1 + hicut))
        Response[condi] <- (i + 1)
    }				
    Response <- round(Response)
    ##------------------------------------------------------------------------------------------------
    
    return(Response)
}

tra = read.csv("data/data_raw/train.csv")
test = read.csv("data/data_raw/test.csv")
cat("prepare train, valid, and test data set...\n")
load("model/rf.RData")
head(ind.train)

load("data/data_preprocess/dt_proprocess_combine.RData")
dt.train.tmp <- dt.preprocessed.combine[isTest == 0]
dt.train.tmp.1 <- merge(dt.train.tmp[, c("Id"), with = F], tra, by.x = "Id", by.y = "Id", all.y = T, sort = F)
tra <- as.data.frame(dt.train.tmp.1)

dt.train <- tra[ind.train, ]
dt.valid <- tra[-ind.train, ]
dt.test <- test

cat("train and train")
response.train <- predict_labels(dt.train, dt.train)
ScoreQuadraticWeightedKappa(dt.train$Response, response.train)
# 0.6470751
cat("train and valid")
response.valid <- predict_labels(dt.train, dt.valid)
ScoreQuadraticWeightedKappa(dt.valid$Response, response.valid)
# 0.640141
cat("train and test")
response.test <- predict_labels(dt.train, dt.test)
# lb 0.64977

Id <- dt.test$Id

outf <- data.frame(Id, Response = response.test)
write.csv(outf, "submit/linear.csv", row.names = F)

## save ##
save(ind.train, response.train, response.valid, response.test, file = "model/lr.RData")
