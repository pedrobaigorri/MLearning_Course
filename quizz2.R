#
# Scripts for the resolution of the QUIZZ 2 for Practical Machine Learning course
#

# QUESTION 1

#Load the Alzheimer's disease data using the commands:

library(AppliedPredictiveModeling)
data(AlzheimerDisease)
library(caret)

# Which of the following commands will create non-overlapping training and test 
# sets with about 50% of the observations assigned to each?

#a)
adData = data.frame(diagnosis,predictors)
train = createDataPartition(diagnosis, p = 0.50,list=FALSE)
test = createDataPartition(diagnosis, p = 0.50,list=FALSE)

#b)
adData = data.frame(diagnosis,predictors)
trainIndex = createDataPartition(diagnosis, p = 0.50)
training = adData[trainIndex,]
testing = adData[-trainIndex,]

#c) -> SOLUTION
adData = data.frame(diagnosis,predictors)
trainIndex = createDataPartition(diagnosis, p = 0.50,list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]



#d)
adData = data.frame(predictors)
trainIndex = createDataPartition(diagnosis,p=0.5,list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]




# QUESTION 2
# Load the cement data using the commands:

library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

# Make a plot of the outcome (CompressiveStrength) versus the index of the samples.
# Color by each of the variables in the data set (you may find the cut2() function 
# in the Hmisc package useful for turning continuous covariates into factors). 
# What do you notice in these plots?
head(training)
training$index <- c(1:nrow(training))

library(Hmisc)

qplot(index, CompressiveStrength, colour = as.factor(Age), data = training)
qplot(index, CompressiveStrength, colour = cut2(Cement, g = 10), data = training)
qplot(index, CompressiveStrength, colour = cut2(BlastFurnaceSlag, g = 10), data = training)
qplot(index, CompressiveStrength, colour = cut2(FlyAsh, g = 10), data = training)
qplot(index, CompressiveStrength, colour = cut2(Water, g = 10), data = training)
qplot(index, CompressiveStrength, colour = cut2(Superplasticizer, g = 10), data = training)
qplot(index, CompressiveStrength, colour = cut2(CoarseAggregate, g = 10), data = training)
qplot(index, CompressiveStrength, colour = cut2(FineAggregate, g = 10), data = training)
qplot(FlyAsh, CompressiveStrength, data = training )


# SOLUTION: There is a non-random pattern in the plot of the outcome versus index 
# that does not appear to be perfectly explained by any predictor suggesting a variable may be missing.


# QUESTION 3
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]


# Make a histogram and confirm the SuperPlasticizer variable is skewed. 
# Normally you might use the log transform to try to make the data more symmetric. 
# Why would that be a poor choice for this variable?
head(training)
training$Superplasticizer + 1.0
qplot(training$Superplasticizer + 1)
table(training$Superplasticizer)
qplot(log10(training$Superplasticizer + 1))
log10(0)
table(log(training$Superplasticizer + 1))



# QUESTION 4
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

#  Find all the predictor variables in the training set that begin with IL. 
#  Perform principal components on these variables with the preProcess() function 
#  from the caret package. 
#  Calculate the number of principal components needed to capture 80% of the variance. How many are there?
head(training)
training2 <- training[ , grepl( "^IL" , names( training ) ) ]
names(training2)

preProc <- preProcess(training2, method="pca",thresh = 0.8)
training2_PC <- predict(preProc, training2)
dim(training2_PC)[2] # 10


# QUESTION 5
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]


# Create a training data set consisting of only the predictors with variable names 
# beginning with IL and the diagnosis. Build two predictive models, one using the 
# predictors as they are and one using PCA with principal components explaining 80% 
# of the variance in the predictors. Use method="glm" in the train function.
names(training)

training2 <- training[ ,  grepl( "^IL" , names(training)) ]
training2$diagnosis <- training$diagnosis
head(training2)

testing2 <- testing[ ,  grepl( "^IL" , names(testing)) ]
testing2$diagnosis <- testing$diagnosis
head(testing2)


# non pca
modelFit <- train(diagnosis ~., method="glm", data=training2)
confusionMatrix(testing2$diagnosis, predict(modelFit,testing2))

# PCA

preProc <- preProcess(training2, method="pca", thresh = 0.8)
training2_PC <- predict(preProc, training2)
testing2_PC <- predict(preProc, testing2)


modelFit_PC <- train(diagnosis ~., method="glm", data=training2_PC)
confusionMatrix(testing2_PC$diagnosis, predict(modelFit_PC,testing2_PC))

summary(modelFit_PC)

####


# SOLUTION:

# Non-PCA Accuracy: 0.65
# PCA Accuracy: 0.72

################################################################################


