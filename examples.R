#Example dummy vars
library(caret)
# check the help file for more details
?dummyVars

customers <- data.frame(
    id=c(10,20,30,40,50),
    gender=c('male','female','female','male','female'),
    mood=c('happy','sad','happy','sad','happy'),
    outcome=c(1,1,0,0,0))


# dummify the data
dmy <- dummyVars(" ~ .", data = customers)
trsf <- data.frame(predict(dmy, newdata = customers))
print(trsf)

##############################################
#
# examples TREES
#
#####
data(iris); library(ggplot2)
names(iris)
table(iris$Species)

inTrain <- createDataPartition(y=iris$Species,
                               p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training); dim(testing)

#qplot(Petal.Width,Sepal.Width,colour=Species,data=training)

library(caret)
modFit <- train(Species ~ .,method="rpart",data=training)
print(modFit$finalModel)

plot(modFit$finalModel, uniform=TRUE, 
     main="Classification Tree")
text(modFit$finalModel, use.n=TRUE, all=TRUE, cex=.8)


library(rattle)
fancyRpartPlot(modFit$finalModel)
