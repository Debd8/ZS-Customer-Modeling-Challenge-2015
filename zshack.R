## Reading the Training data into R

zs_train <- read.csv("C:/Users/Debarati/Desktop/Zsassociates_Ideatory/TrainingData.csv",header = T, stringsAsFactors = F)

## Removing the observations where the target variable is missing

zs_train <- subset(zs_train, Labels != "NA")

## Summarizing missing data for all variables

propmiss <- function(dataframe) 
  {
  m <- sapply(dataframe, function(x) 
    {
    data.frame(nmiss=sum(is.na(x)),n=length(x),propmiss=sum(is.na(x))/length(x))
  })
  d <- data.frame(t(m))
  d <- sapply(d, unlist)
  d <- as.data.frame(d)
  d$variable <- row.names(d)
  row.names(d) <- NULL
  d <- cbind(d[ncol(d)],d[-ncol(d)])
  return(d[order(d$propmiss), ])
}
zs_missing <- propmiss(zs_train)

## Variables with less than 50% missing data are only retained in the dataset

keep_var <- subset(zs_missing, propmiss <= 0.5, select = variable)
zs_train_final <- zs_train[,names(zs_train) %in% keep_var[,1]]

## Creating a new data frame after removing the variables "Customer.ID" & "Labels"

drop <- c("Customer.ID", "Labels")
zs.rank <- zs_train_final[,!names(zs_train_final) %in% drop]

## Principle component analysis for dimension reduction
## Calculating the maximum likelihood estimates of the variance-covariance matrix of zs.rank

library(mvnmle)
covmat <- (mlest(zs.rank))$sigmahat

## Calcuting the eigenvalues and corresponding eigenvectors of the variance-covariance matrix

library(Matrix)
e <- eigen(covmat)

## Identifying the eigenvector corresponding to minimum eigenvalue

last.vector <- e$vectors[,which.min(e$values)]
last.vector[order(abs(last.vector))]

## Removing the variables with the highest loadings on last.vector (variables with highest absolute value of last.vector)

d <- c("Variable_105", "Variable_91", "Variable_37", "Variable_189")
zs_small <- zs_train_final[,!names(zs_train_final) %in% d]

## Applying Amelia algorithm with 10 iterations for missing data imputation and writing the output to a csv file

library(Amelia)
set.seed(123)
am.zs <- amelia(x=zs_small, idvars = c("Labels","Customer.ID"), m=10)
summary(am.zs)
write.amelia(am.zs,file.stem = "zs_outdata",format="csv")

## Reading the final imputed output dataset into R

zs_imputed <- read.csv("C:/Users/Debarati/Desktop/zs_outdata10.csv",header = T, stringsAsFactors = F, strip.white = T)
zs_imputed <- zs_imputed[,-1]

## Defining a variable transformation function where variables have negative values

transform <- function(x)
{
  sign(x) * abs(x)^(1/3)
}

## Applying the transform function to imputed dataset

zs_imputed[,2:38] <- as.data.frame(apply(zs_imputed[,2:38], 2, transform))

## Converting the Target variable of the final transformed dataset to categorical/factor

zs_imputed$Labels <- as.factor(zs_imputed$Labels)

## Fitting a random forest model

library(randomForest)
set.seed(123)
zs.rf <- randomForest(Labels~.-Customer.ID, data = zs_imputed, ntree = 500, importance = T, do.trace=100)

## Fitting a svm model

library(e1071)
set.seed(123)
zs.svm <- svm(Labels~.-Customer.ID, data = zs_imputed, probability = T)

## Reading the Evaluation data into R

zs_eval <- read.csv("C:/Users/Debarati/Desktop/Zsassociates_Ideatory/EvaluationData.csv",header = T, stringsAsFactors = F)
zs_eval$Labels <- NULL

## Retaining the same variables as in the training set

zs_eval_final <- zs_eval[,names(zs_eval) %in% keep_var[,1]]
zs_eval_small <- zs_eval_final[,!names(zs_eval_final) %in% d]

## Applying Amelia algorithm with 10 iterations for missing data imputation and writing the output to a csv file

library(Amelia)
set.seed(123)
am.zs.eval <- amelia(x=zs_eval_small, idvar = "Customer.ID", m=10)
summary(am.zs.eval)
write.amelia(am.zs.eval,file.stem = "zs_eval_outdata",format="csv")

## Reading the final imputed output dataset into R

zs_eval_imputed <- read.csv("C:/Users/Debarati/Desktop//Zsassociates_Ideatory/zs_eval_outdata10.csv",header = T, stringsAsFactors = F, strip.white = T)
zs_eval_imputed <- zs_eval_imputed[,-1]

## Applying the transform function to imputed dataset

zs_eval_imputed[,2:38] <- as.data.frame(apply(zs_eval_imputed[,2:38], 2, transform))

## Predicting on the evaluation set

zs_predict_rf <- predict(zs.rf, zs_eval_imputed[,2:38], type='prob')
zs_predict_svm <- predict(zs.svm, zs_eval_imputed[,2:38], probability = T)

## Creating an ensemble model

predict.final <- (0.45 * zs_predict_rf + 0.55 * zs_predict_svm)

## Converting the class probabilities to class labels

predicted_Labels <- ifelse(predict.final > 0.4, 1, 0)
zs_eval_imputed <- cbind(zs_eval_imputed, predicted_Labels)

## Final data for submission

final_submission <- zs_eval_imputed[,c(1,39)]