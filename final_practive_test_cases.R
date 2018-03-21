#
# Dirty Code for final practice Machine Learning final practice and 
# test cases
#
# Pedro A. Alonso Baigorri
#

library(caret)
library(RANN)


# Getting the data
{
    trainF <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    testF <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    
    if (!file.exists("./data/pml-training.csv"))
    {
        if (!file.exists("./data")){dir.create("./data")}
        download.file(trainF, "./data/pml-training.csv")
    }
    
    if (!file.exists("./data/pml-testing.csv"))
    {
        if (!file.exists("./data")){dir.create("./data")}
        download.file(testF, "./data/pml-testing.csv")
    }
}

# Opening data
train <- read.table(trainF, header = TRUE, sep = ",")
test <- read.table(testF, header = TRUE, sep = ",")

# testing data
head(train)
head(test)
dim(train)
dim(test)

#removing time, user name and other variables out of interest
train <- train[, -c(1:7)]
test <- test[, -c(1:7)]


#analyzing variance in train and test
var <- nearZeroVar(train, saveMetrics=TRUE)
table(var$nzv)

var <- nearZeroVar(test, saveMetrics=TRUE)
table(var$nzv)

# removing variables of near zero variance at test
trainB <- train[, -nearZeroVar(test)]
testB <- test[, -nearZeroVar(test)]

# setup of models
set.seed(300)
train_control<- trainControl(method="cv", number=10, savePredictions = TRUE)

#train models decision tree
fitTree <- train(classe ~ .,  data = trainB, method="rpart", trControl=train_control)
print(fitTree)
plot(fitTree)


#train models C5
fitC5 <- train(classe ~ .,  data = trainB, method="C5.0",  trControl=train_control)
print(fitC5)
plot(fitC5)

# in of sample errors
pred <- predict(fitC5)
IOS_errors_c5 <- sum(pred != trainB$classe)
IOS_errors_c5_rate <- IOS_errors_c5 / nrow(trainB)


# out of sample errors
pred <- subset(fitC5$pred, trials == 20 & model == "rules" & winnow == FALSE )
OOS_errors_c5 <- tapply(pred$pred != pred$obs, pred$Resample, sum)
n <- table(pred$Resample)
OOS_errors_rate_c5 <- mean(OOS_errors_c5/n)
print(round(OOS_errors_rate_c5, 3))



#train models GBM
fitGbm <- train(classe ~ .,  data = trainB, method="gbm", verbose = FALSE, trControl=train_control)
print(fitGbm)
plot(fitGbm)


# in of sample errors
pred <- predict(fitGbm)
IOS_errors_gbm <- sum(pred != trainB$classe)
IOS_errors_gbm_rate <- IOS_errors_gbm / nrow(trainB)


# out of sample errors
pred <- subset(fitGbm$pred, n.trees == 150 & interaction.depth == 3)
OOS_errors_gbm <- tapply(pred$pred != pred$obs, pred$Resample, sum)
n <- table(pred$Resample)
OOS_errors_rate_gbm <- mean(OOS_errors_gbm/n)
print(round(OOS_errors_rate_gbm, 3))


sum(fitGbm$resample["Accuracy"])/10



#prediction test
prediction <- predict(fitGbm, newdata = testB)
prediction2 <- predict(fitC5, newdata = testB)
print(prediction)
print(prediction2)
