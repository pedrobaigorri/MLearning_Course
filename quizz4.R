#
# Scripts for the resolution of the QUIZZ 4 for Practical Machine Learning course
#

#
# QUESTION 1
#
# Load the vowel.train and vowel.test data sets:

library(ElemStatLearn)
library(caret)

data(vowel.train)

data(vowel.test)
    
# Set the variable y to be a factor variable in both the training and test set. 
# Then set the seed to 33833. Fit (1) a random forest predictor relating the 
# factor variable y to the remaining variables and (2) a boosted predictor 
# using the "gbm" method. Fit these both with the train() command in the caret package.

# What are the accuracies for the two approaches on the test data set? 
# What is the accuracy among the test set samples where the two methods agree?

vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)

table(vowel.train$y)
table(vowel.test$y)

set.seed(33833)

# random
fitRandom <- train(y ~ .,  data = vowel.train, method="rf")
print(fitRandom$finalModel)

predictRandom <- predict(fitRandom, newdata = vowel.test)
length(predictRandom)

confusionMatrix(predictRandom, vowel.test$y)$overall["Accuracy"]

#gbm
fitGbm <- train(y ~ .,  data = vowel.train, method="gbm", verbose = FALSE)
print(fitGbm$finalModel)

predictGbm <- predict(fitGbm, newdata = vowel.test)
length(predictGbm)

confusionMatrix(predictGbm, vowel.test$y)$overall["Accuracy"]

#agreement accuracy
rf_ok <- ifelse(predictRandom == vowel.test$y, 1, 0)
sum(rf_ok)/length(rf_ok)

gbm_ok <- ifelse(predictGbm == vowel.test$y, 1, 0)
sum(gbm_ok)/length(gbm_ok)

agreed <- gbm_ok * rf_ok

sum(agreed)/length(agreed)


#
# QUESTION 2
#
# Load the Alzheimer's data using the following commands

library(caret)

library(gbm)

set.seed(3433)

library(AppliedPredictiveModeling)

data(AlzheimerDisease)

adData = data.frame(diagnosis,predictors)

inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]

training = adData[ inTrain,]

testing = adData[-inTrain,]

# Set the seed to 62433 and predict diagnosis with all the other variables 
# using a random forest ("rf"), boosted trees ("gbm") and linear discriminant 
# analysis ("lda") model. Stack the predictions together using random forests ("rf"). 
# What is the resulting accuracy on the test set? Is it better or worse than each 
# of the individual predictions?


set.seed(62433)

rf <- train(diagnosis ~ .,  data = training, method="rf", prox=TRUE)
gbm <- train(diagnosis ~ .,  data = training, method="gbm", verbose=FALSE)
lda <- train(diagnosis ~ .,  data = training, method="lda")

rf_p <- predict(rf, newdata=testing)
gbm_p <- predict(gbm, newdata=testing)
lda_p <- predict(lda, newdata=testing)

confusionMatrix(rf_p, testing$diagnosis) # Accuracy : 0.7683 
confusionMatrix(gbm_p, testing$diagnosis) # Accuracy : 0.8171
confusionMatrix(lda_p, testing$diagnosis) # Accuracy: 0.7683

data_stacked <- data.frame(rf=predict(rf), gbm=predict(gbm), lda=predict(lda), diagnosis = training$diagnosis)
rf_stack <- train(diagnosis ~ .,  data = data_stacked, method="rf", prox=TRUE)
print(rf_stack)

data_stacked_p <- data.frame(rf=rf_p, gbm=gbm_p, lda=lda_p, diagnosis = testing$diagnosis)
rf_stacked_p <- predict(rf_stack, newdata=data_stacked_p)
confusionMatrix(rf_stacked_p, testing$diagnosis) # Accuracy : 0.8171

# Stacked Accuracy: 0.80 is better than random forests and lda and the same as boosting.

# QUESTION 3 -> CEMENT
# 
# Load the concrete data with the commands:
    
set.seed(3523)

library(caret)
library(AppliedPredictiveModeling)

data(concrete)

inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]

training = concrete[ inTrain,]

testing = concrete[-inTrain,]



# Set the seed to 233 and fit a lasso model to predict Compressive Strength. 
# Which variable is the last coefficient to be set to zero as the penalty 
# increases? (Hint: it may be useful to look up ?plot.enet).
library(elasticnet)

head(training)
??plot.enet
fit <- train(CompressiveStrength ~ ., data = concrete, method ="lasso")
plot(fit$finalModel, xvar="penalty", use.color=T)

# QUESTION 4 -> 96

# Load the data on the number of visitors to the instructors blog from here:
    
#https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv

# Using the commands:

library(lubridate) # For year() function below

setwd("D://Pedro//TID//BI4TD//DATA SCIENCE//COURSERA//8 - Practical Machine Learning")

fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv"

download.file(fileURL, "gaData.csv")


dat = read.csv("gaData.csv")

training = dat[year(dat$date) < 2012,]

testing = dat[(year(dat$date)) > 2011,]

tstrain = ts(training$visitsTumblr)
tstest = ts(testing$visitsTumblr)

# Fit a model using the bats() function in the forecast package to the training 
# time series. Then forecast this model for the remaining time points. For how
# many of the testing points is the true value within the 95% prediction interval bounds?

library(forecast)
#??bats()

fit <- bats(tstrain)
plot(forecast(fit))

dim(training)
dim(testing)
prediction <- forecast(fit, h = nrow(testing))

length(prediction$mean)
print(tstrain)
print(tstest)

forecastObj <- prediction
betweenVal <- sum(testing$visitsTumblr > forecastObj$lower &  testing$visitsTumblr < forecastObj$upper)
betweenVal / nrow(testing) * 100 #solution

tstest[1:10]
prediction




head(dat)
plot(dat$date, dat$visitsTumblr)

# QUESTION 5 -> 6.72

# Load the concrete data with the commands:
    
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]


# Set the seed to 325 and fit a support vector machine using the e1071 
# package to predict Compressive Strength using the default settings. 
# Predict on the testing set. What is the RMSE?

set.seed(325)
library("e1071")
model <- svm(CompressiveStrength ~ ., data = training )

prediction <- predict(model, testing)
accSvm <- accuracy(prediction, testing$CompressiveStrength)
data.frame(accSvm)["RMSE"]

# ROC CURVE AND AUC EXAMPLES
library(ElemStatLearn)
library(caret)

data(vowel.train)

data(vowel.test)

vowel.train$class <- as.factor(ifelse(vowel.train$y == 1, "X1.", "X0."))
vowel.test$class <- as.factor(ifelse(vowel.test$y == 1, "X1.", "X0."))


table(vowel.train$class)
table(vowel.test$class)

set.seed(33833)

#gbm
head(vowel.train)

ctrl <- trainControl(classProbs = TRUE)

fitGbm <- train(class ~ .,  data = vowel.train, method="gbm", verbose = FALSE, metric = "ROC", trControl = ctrl)

make.names(levels(vowel.train$class))
print(fitGbm$finalModel)

predictGbm <- predict(fitGbm, newdata = vowel.test, type="prob")
length(predictGbm)

lift_results <- data.frame(Class = vowel.test$class)
lift_results$gbm <- predictGbm[, "X1."]

head(lift_results)
head(predictGbm)[, "X1."]


lift_obj <- lift(Class ~ gbm, data = lift_results)
plot(lift_obj, values = 60, auto.key = list(columns = 3,
                                            lines = TRUE,
                                            points = FALSE))

library(pROC)
roc_obj <- roc(lift_results$Class, lift_results$gbm)
auc(roc_obj)
