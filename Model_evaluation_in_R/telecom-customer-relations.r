#
# ---------- READ AND ARRANGE DATA ----------
#

# Read the file containing the independent variables.
# Treat 'NA' and empty strings in the file as missing data.
rawData <- read.table("A:\\PROGRAMMING\\R\\module2\\hw2\\telecom-customer-relations\\telecom_small_train.data.gz",
    header=TRUE,
    sep='\t',
    na.strings=c('NA',''), encoding = "UTF-8")

# Read the "Churn" dependent variable.
churn <- read.table("A:\\PROGRAMMING\\R\\module2\\hw2\\telecom-customer-relations\\telecom_small_train_churn.labels.txt",
    header=FALSE,
    sep='\t', encoding = "UTF-8")

# Append "Churn" as a new column
rawData$churn <- churn$V1

# Read and append the "Appetency" dependent variable.
appetency <- read.table("A:\\PROGRAMMING\\R\\module2\\hw2\\telecom-customer-relations\\telecom_small_train_appetency.labels.txt",
    header=FALSE,
    sep='\t')
rawData$appetency <- appetency$V1 

# Read and append the "Upselling" dependent variable.
upselling <- read.table("A:\\PROGRAMMING\\R\\module2\\hw2\\telecom-customer-relations\\telecom_small_train_upselling.labels.txt",
    header=FALSE,
    sep='\t')
rawData$upselling <- upselling$V1

# Make the random number generator reproducible on each run.
set.seed(248962)

# Split data into train and test subsets.
rawData$rgroup <- runif(dim(rawData)[[1]])
dTrainAll <- subset(rawData, rgroup <= 0.9)
dTest <- subset(rawData, rgroup > 0.9)

# Further split training data into training and calibration.
useForCal <- rbinom(n=dim(dTrainAll)[[1]], size = 1, prob = 0.5) > 0 
dCal <- subset(dTrainAll, useForCal)
dTrain <- subset(dTrainAll, !useForCal)

# Remember which variables are independent and dependent.
outcomes=c('churn', 'appetency', 'upselling')
vars <- setdiff(colnames(dTrainAll), c(outcomes, 'rgroup'))

# Identify which variables are categorical.
catVars <- vars[sapply(dTrainAll[, vars], class) %in% c('factor', 'character')] 

# Identify which variables are numeric.
numericVars <- vars[sapply(dTrainAll[, vars], class) %in% c('numeric', 'integer')] 

# Remove the obsolete variables from the workspace.
rm(list=c('rawData', 'churn', 'appetency', 'upselling'))

# Choose which of the three outcomes to model (Churn in this case). 
outcome <- 'churn'

# Positive outcome is '1', negative is '-1' in the files.
pos <- '1'

#
# ---------- CLEAN DATA AND SELECT VARIABLES ----------
#

# Each variable we use represents a chance of explaining
# more of the outcome variation (a chance of building a better
# model) but also represents a possible source of noise and
# overfitting. To control this effect, we often preselect
# which subset of variables we’ll use to fit. Variable
# selection can be an important defensive modeling step even
# for types of models that "don’t need it". The following code shows a
# hand-rolled variable selection loop where each variable is
# scored according to a deviance inspired score, where a
# variable is scored with a bonus proportional to the change
# in in scaled log likelihood of the training data. The
# score is a bit ad hoc, but tends to work well in selecting
# variables. Notice we’re using performance on the calibration
# set (not the training set) to pick variables. Note that we
# don’t use the test set for calibration; to do so lessens the
# reliability of the test set for model quality confirmation.

library('ROCR')

# AUC - Area Under (ROC) Curve
calcAUC <- function(predcol, outcol) {
    perf <- performance(prediction(predcol, outcol == pos), 'auc')
    as.numeric(perf@y.values)
}

# Scoring categorical variables by AUC.

# Given a vector of training outcomes (outCol), a categorical training variable (varCol),
# and a prediction variable (appCol), use outCol and varCol to build a single-variable
# model and then apply the model to appCol to get new predictions. 
mkPredC <- function(outCol, varCol, appCol) {
    # Get stats on how often outcome is positive during training. 
    pPos <- sum(outCol == pos) / length(outCol) 
   
    # Get stats on how often outcome is positive for NA values of variable during training.
    naTab <- table(as.factor(outCol[is.na(varCol)]))
    pPosWna <- (naTab / sum(naTab))[pos] 
    vTab <- table(as.factor(outCol), varCol)
   
    # Get stats on how often outcome is positive, conditioned on levels of training variable.
    pPosWv <- (vTab[pos, ] + 1.0e-3 * pPos) / (colSums(vTab) + 1.0e-3) 
   
    # Make predictions by looking up levels of appCol.
    pred <- pPosWv[appCol]
   
    # Add in predictions for NA levels ofappCol.
    pred[is.na(appCol)] <- pPosWna
   
    # Add in predictions for levels of appCol that weren’t known during training.
    pred[is.na(pred)] <- pPos
    pred 
}

for (v in catVars) {
    pi <- paste('pred', v, sep='')
    dTrain[, pi] <- mkPredC(dTrain[, outcome], dTrain[, v], dTrain[, v])
    dCal[, pi] <- mkPredC(dTrain[, outcome], dTrain[, v], dCal[, v])
    dTest[, pi] <- mkPredC(dTrain[, outcome], dTrain[, v], dTest[, v])
}

# Scoring numeric variables by AUC.
mkPredN <- function(outCol, varCol, appCol) {
    nval <- length(unique(varCol[!is.na(varCol)]))
    if(nval <= 1) {
        pPos <- sum(outCol == pos) / length(outCol)
        return(pPos + numeric(length(appCol)))
    }
    cuts <- unique(as.numeric(quantile(varCol, probs=seq(0, 1, 0.1), na.rm=TRUE)))
    varC <- cut(varCol, cuts)
    appC <- cut(appCol, cuts)
    mkPredC(outCol, varC, appC)
}

for (v in numericVars) {
    pi <- paste('pred', v, sep='')
    dTrain[, pi] <- mkPredN(dTrain[, outcome], dTrain[, v], dTrain[, v])
    dTest[, pi] <- mkPredN(dTrain[, outcome], dTrain[, v], dTest[, v])
    dCal[, pi] <- mkPredN(dTrain[, outcome], dTrain[, v], dCal[, v])
    aucTrain <- calcAUC(dTrain[, pi], dTrain[, outcome])
    if (aucTrain >= 0.55) {
        aucCal <- calcAUC(dCal[, pi], dCal[, outcome])
        print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f", pi, aucTrain, aucCal))
    }
}

logLikelihood <- function(outCol, predCol) { 
  sum(ifelse(outCol == pos, log(predCol), log(1 - predCol)))
}

selVars <- c()
minStep <- 5
baseRateCheck <- logLikelihood(dCal[, outcome],
   sum(dCal[, outcome] == pos) / length(dCal[, outcome]))

# Run through categorical variables and pick 
# based on a deviance improvement (related to 
# difference in log likelihood.

for (v in catVars) { 
    pi <- paste('pred', v, sep='')
    liCheck <- 2 * ((logLikelihood(dCal[, outcome], dCal[, pi]) - baseRateCheck))
    if (liCheck > minStep) {
        print(sprintf("%s, calibrationScore: %g", pi, liCheck))
        selVars <- c(selVars, pi)
    }
}

# Run through numeric variables and pick based on a deviance improvement. 
for (v in numericVars) {
    pi <- paste('pred', v, sep='')
    liCheck <- 2 * ((logLikelihood(dCal[, outcome], dCal[, pi]) - baseRateCheck))
    
    if (liCheck >= minStep) {
        print(sprintf("%s, calibrationScore: %g", pi, liCheck))
        selVars <- c(selVars, pi)
    }
}

#
# ---------- PREDICT OUTCOME ----------
#

formulaString <- paste(outcome, ' > 0 ~ ', paste(selVars, collapse=' + '), sep='')

# Example: fitting a logistic regression model.
logisticModel <- glm(as.formula(formulaString),
    data=dTrain,
    family=binomial(link='logit'))

print(calcAUC(predict(logisticModel, newdata=dTrain), dTrain[, outcome]))
print(calcAUC(predict(logisticModel, newdata=dCal), dCal[, outcome]))
print(calcAUC(predict(logisticModel, newdata=dTest), dTest[, outcome]))

########## PRACTICAL EXERCISE ###########

# Fitting a Random Forest model:
library(randomForest)
rf <-randomForest(as.formula(formulaString),data=dTrain, ntree=20, 
                        keep.forest=TRUE, importance=TRUE)
importances <- rf$importance
print(calcAUC(predict(rf, newdata=dTrain), dTrain[, outcome]))
print(calcAUC(predict(rf, newdata=dCal), dCal[, outcome]))
print(calcAUC(predict(rf, newdata=dTest), dTest[, outcome]))

# Fitting Naive Bayes:
library(e1071)
nb <- naiveBayes(as.formula(formulaString), data = dTrain)
print(calcAUC(predict(nb, newdata=dTrain, type = "raw")[,2], dTrain[, outcome]))
print(calcAUC(predict(nb, newdata=dCal, type = "raw")[,2], dCal[, outcome]))
print(calcAUC(predict(nb, newdata=dTest, type = "raw")[,2], dTest[, outcome]))

# Fitting SVM
sv <- svm(as.formula(formulaString), data = dTrain, type = "C-classification", kernel = "linear", probability=TRUE)
print(calcAUC(attr(predict(sv, newdata = dTrain, probability = TRUE), "probabilities")[,2], dTrain[, outcome]))
print(calcAUC(attr(predict(sv, newdata = dCal, probability = TRUE), "probabilities")[,2], dCal[, outcome]))
print(calcAUC(attr(predict(sv, newdata = dTest, probability = TRUE), "probabilities")[,2], dTest[, outcome]))

## Let's combine training and calibration sets to have more variables for training,
## because our main task is to build a graph between 'number of observations for training' and 'AUC'.
## In general, calibration set could be used to calculate average AUC using cross-validation.
data_train <- rbind2(dTrain,dCal)
## shuffle matrix:
data_train <- data_train[sample(nrow(data_train)),]
AUC_rf_20 <- c()
AUC_rf_40 <- c()
AUC_nb <- c()
AUC_sv <- c()
obs_nums = c(5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 44900)
for (obs_num in obs_nums) {
  ## random forest, 20 trees:
  rf <- randomForest(as.formula(formulaString),data=data_train[1:obs_num,], ntree=20, 
                    keep.forest=TRUE, importance=TRUE)
  AUC_rf_20 <- c(AUC_rf, (calcAUC(predict(rf, newdata=dTest), dTest[, outcome])))
  
  ## random forest, 40 trees:
  rf <- randomForest(as.formula(formulaString),data=data_train[1:obs_num,], ntree=40, 
                     keep.forest=TRUE, importance=TRUE)
  AUC_rf_40 <- c(AUC_rf_40, (calcAUC(predict(rf, newdata=dTest), dTest[, outcome])))
  
  ## naive bayes:
  nb <- naiveBayes(as.formula(formulaString), data = data_train[1:obs_num,])
  AUC_nb <- c(AUC_nb, (calcAUC(predict(nb, newdata=dTest, type = "raw")[,2], dTest[, outcome])))

  ## svm:
  sv <- svm(as.formula(formulaString), data = data_train[1:obs_num,], type = "C-classification", kernel = "linear", probability=TRUE)
  AUC_sv <- c(AUC_sv ,(calcAUC(attr(predict(sv, newdata = dTest, probability = TRUE), "probabilities")[,2], dTest[, outcome])))
}

#> AUC_rf_20
#[1] 0.5951829 0.6443515 0.6335322 0.6310509 0.6409053 0.6490711 0.6594395 0.6523453 0.6548144

#> AUC_rf_40
#[1] 0.6243909 0.6245810 0.6464440 0.6556309 0.6649201 0.6587690 0.6690365 0.6575723 0.6650825

#> AUC_nb
#[1] 0.6470844 0.6439871 0.6424710 0.6426372 0.6412745 0.6418172 0.6423696 0.6424814 0.6416819

#> AUC_sv
#[1] 0.4586925 0.5229277 0.4897654 0.5515314 0.5001234 0.4623107 0.5689618 0.4970326 0.4425339

##  Visualization:
AUC_sv <- round(AUC_sv, digits=5)
AUC_nb <- round(AUC_nb, digits=5)
AUC_rf_20 <- round(AUC_rf_20, digits=5)
AUC_rf_40 <- round(AUC_rf_40, digits=5)

g_range <- range(0, AUC_rf_20, AUC_rf_40, AUC_nb, AUC_sv)
plot(AUC_rf_20, type="o", col="blue", ylim=c(0.3, g_range[2]+0.2), axes=FALSE, ann=FALSE)
axis(1, at=1:9, lab=c("0.5K","1K","1.5K","2K","2.5K","3K","3.5K","4K","4.49K"))
axis(2, las=1, at=0.05*0:15)
box()
lines(AUC_rf_40, type="o", pch=22, lty=2, col="black")
lines(AUC_nb, type="o", pch=23, lty=3, col="red")
lines(AUC_sv, type="o", pch=24, lty=4, col="brown")
title(main="Evaluation of models", col.main="black", font.main=4)
title(xlab="Number of observations", col.lab=rgb(0,0.5,0))
title(ylab="AUC", col.lab=rgb(0,0.5,0))
legend(0.8, g_range[2]+0.2, c("Random Forest_20", "Random Forest_40", "Naive Bayes", "SVM"), cex=0.9,
       col=c("blue", "black", "red", "brown"), pch=21:24, lty=1:4, bty='n')




