#
# Scripts for the resolution of the QUIZZ 3 for Practical Machine Learning course
#

# QUESTION 1

# Load the cell segmentation data from the AppliedPredictiveModeling package using the commands:
    
library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)

# 1. Subset the data to a training set and testing set based on the Case variable in the data set.

# 2. Set the seed to 125 and fit a CART model with the rpart method using all predictor variables and default caret settings.

# 3. In the final model what would be the final model prediction for cases with the following variable values:
    
    # a. TotalIntench2 = 23,000; FiberWidthCh1 = 10; PerimStatusCh1=2

    # b. TotalIntench2 = 50,000; FiberWidthCh1 = 10;VarIntenCh4 = 100

    # c. TotalIntench2 = 57,000; FiberWidthCh1 = 8;VarIntenCh4 = 100

    # d. FiberWidthCh1 = 8;VarIntenCh4 = 100; PerimStatusCh1=2



head(segmentationOriginal)

training <- subset(segmentationOriginal, Case == "Train")
testing <- subset(segmentationOriginal, Case == "Test")

table(segmentationOriginal$Case)

dim(training); dim(testing);dim(segmentationOriginal)

set.seed(125)

modFit <- train(Class ~ .,method="rpart",data=training)
print(modFit$finalModel)

plot(modFit$finalModel, uniform=TRUE, 
     main="Classification Tree")
text(modFit$finalModel, use.n=TRUE, all=TRUE, cex=.8)

table(training$Class)


copy_testing <- testing[1,]

copy_testing[1, "TotalIntench2"] <- 23000
copy_testing[1, "FiberWidthCh1"] <- 10
copy_testing[1, "PerimStatusCh1"] <- 2

predict(modFit,newdata=copy_testing)

copy_testing[1, "TotalIntench2"] <- 50000
copy_testing[1, "FiberWidthCh1"] <- 10
copy_testing[1, "VarIntenCh4"] <- 100

predict(modFit,newdata=copy_testing)

copy_testing[1, "TotalIntench2"] <- 57000
copy_testing[1, "FiberWidthCh1"] <- 8
copy_testing[1, "VarIntenCh4"] <- 100

predict(modFit,newdata=copy_testing)

copy_testing[1, "FiberWidthCh1"] <- 8
copy_testing[1, "VarIntenCh4"] <- 100
copy_testing[1, "PerimStatusCh1"] <- 2

predict(modFit,newdata=copy_testing)

# El predict no funciona. Hay que hacerlo a mano:

PS/WS/PS/ impossible



# QUESTION 2

# If K is small in a K-fold cross validation is the bias in the estimate of 
# out-of-sample (test set) accuracy smaller or bigger? If K is small is the 
# variance in the estimate of out-of-sample (test set) accuracy smaller or bigger. 
# Is K large or small in leave one out cross validation?

# The bias is larger and the variance is smaller. Under leave one out cross validation K is equal to the sample size.


# QUESTION 3

# Load the olive oil data using the commands:
    

load("C://Users//paab//Downloads//olive_data//olive.rda")

head(olive)
olive = olive[,-1]
head(olive)

# (NOTE: If you have trouble installing the pgmm package, you can download the 
# -code-olive-/code- dataset here: olive_data.zip. After unzipping the archive, 
# you can load the file using the -code-load()-/code- function in R.)

# These data contain information on 572 different Italian olive oils from multiple 
# regions in Italy. Fit a classification tree where Area is the outcome variable. 
# Then predict the value of area for the following data frame using the tree command with all defaults



newdata = as.data.frame(t(colMeans(olive)))

# What is the resulting prediction? Is the resulting prediction strange? Why or why not?

library(caret)
library(AppliedPredictiveModeling)

set.seed(3433)

inTrain = createDataPartition(olive$Area, p = 3/4)[[1]]
training = olive[ inTrain,]
testing = olive[-inTrain,]

modFit <- train(Area ~ .,method="rpart",data=training)
print(modFit$finalModel)

plot(modFit$finalModel, uniform=TRUE, 
     main="Classification Tree")
text(modFit$finalModel, use.n=TRUE, all=TRUE, cex=.8)

predict(modFit,newdata=newdata)

# Solution: 2.783. It is strange because Area should be a qualitative variable - 
# but tree is reporting the average value of Area as a numeric variable in the leaf predicted for newdata

# QUESTION 4
# Load the South Africa Heart Disease Data and create training and test sets with the following code:

# Then set the seed to 13234 and fit a logistic regression model 
# (method="glm", be sure to specify family="binomial") with Coronary Heart Disease 
# (chd) as the outcome and age at onset, current alcohol consumption, obesity levels, 
# cumulative tabacco, type-A behavior, and low density lipoprotein cholesterol as predictors. 
# Calculate the misclassification rate for your model using this function and a 
# prediction on the "response" scale:    

library(ElemStatLearn)
data(SAheart)
library(caret)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]



set.seed(13234)
head(SAheart)

trainSA <- trainSA[, c(-1, -4, -5 )]
testSA <- testSA[, c(-1, -4, -5 )]
head(trainSA)    

modFit<- train(chd ~ ., data = trainSA, method="glm", family ="binomial")
head(trainSA)

missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values
)/length(values)}



# What is the misclassification rate on the training set? 
missClass(trainSA$chd, predict(modFit, newdata = trainSA))

# What is the misclassification rate on the test set?
missClass(testSA$chd, predict(modFit, newdata = testSA))

#Test Set Misclassification: 0.31
# Training Set: 0.27


# QUESTION 5
# Load the vowel.train and vowel.test data sets:
    
    
library(ElemStatLearn)
data(vowel.train)
data(vowel.test)

# Set the variable y to be a factor variable in both the training and test set. 
# Then set the seed to 33833. Fit a random forest predictor relating the factor 
# variable y to the remaining variables. 
# Read about variable importance in random forests here: 
# http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr 
# The caret package uses by default the Gini importance.

# Calculate the variable importance using the varImp function in the caret package. What is the order of variable importance?
    
# [NOTE: Use randomForest() specifically, not caret, as there's been some issues reported with that approach. 11/6/2016]

library(randomForest)
head(vowel.train)
vowel.train$y <- as.factor(vowel.train$y)

set.seed(33833)

rFit <- randomForest(y ~ ., vowel.train)
imp <- importance(rFit)
order(-imp)


