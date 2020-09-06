#install.packages("caret")
library(caret)

set.seed(123) #randomization`

# Load Data

data <- read.csv(file.choose())

# Normalize Data
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

data_n <- as.data.frame(lapply(data[-c(1,95)], normalize))

data_n$target <- data$target

str(data_n)

#creating train & test data partition

trainIndex <- createDataPartition(data_n$target,p=0.75,list=FALSE)

head(trainIndex)

data_train <- data_n[trainIndex,]

data_test <- data_n[-trainIndex,]

# Class balance info

table(data_train$target)

table(data_test$target)

# Check for Factorability using KMO test - For Dimension Redn
# Optional

#install.packages("psych")
library(psych)

KMO(cor(data[-c(1,95)]))

# Perform Dimension Reduction - optional

X <- as.matrix(data[-c(1,95)])

PCA <- princomp(~X, scores = TRUE, cor=TRUE)

summary(PCA)

# Perform Multinomial Logistic Regression

library(nnet)

multinom <- multinom(target ~ ., data = data_train[-1,], maxit=3000)


predicted=predict(multinom,data_test,type="class")

predicted

# Evaluate Model performance
install.packages("e1071")
library(e1071)
library(caret)
confusionMatrix(predicted,
                data_test$target,
                mode="everything")

# Overall Accuracy = 76.58%

# Perform LDA
#install.packages("DiscriMiner")
library(DiscriMiner)

# Split data into DV & IV's
X <- data_train[,-94]
Y <- data_train[,94]

Mahalanobis = linDA(X,Y)

Mahalanobis


predicted=classify(Mahalanobis,data_test[,-94])

str(predicted)

predicted <- predicted$pred_class

# Evaluate Model performance
install.packages("e1071")
library(e1071)
library(caret)
confusionMatrix(predicted,
                data_test$target,
                mode="everything")

# Overall Accuracy = 70.48%



# Perform SVM

library(e1071)        # For SVM Model

#Linear kernel
supvm = svm(target ~ ., data=data_train , kernel = "linear")

predicted = predict(supvm,newdata = data_test)

# Evaluate Model performance
install.packages("e1071")
library(e1071)
library(caret)
confusionMatrix(predicted,
                data_test$target,
                mode="everything")

# Overall Accuracy = 76.97%


#Polynomial Kernel
supvm = svm(target ~ ., data=data_train , kernel = "polynomial", degree=3)

predicted = predict(supvm,newdata = data_test)

# Evaluate Model performance
install.packages("e1071")
library(e1071)
library(caret)
confusionMatrix(predicted,
                data_test$target,
                mode="everything")

# Overall Accuracy = 68.28%


#Sigmoid Kernel
supvm = svm(target ~ ., data=data_train , kernel = "sigmoid")

predicted = predict(supvm,newdata = data_test)

# Evaluate Model performance
install.packages("e1071")
library(e1071)
library(caret)
confusionMatrix(predicted,
                data_test$target,
                mode="everything")

# Overall Accuracy = 62.94%

#Radial Kernel
supvm = svm(target ~ ., data=data_train , kernel = "radial")

predicted = predict(supvm,newdata = data_test)

# Evaluate Model performance
install.packages("e1071")
library(e1071)
library(caret)
confusionMatrix(predicted,
                data_test$target,
                mode="everything")

# Overall Accuracy = 78.56%


# Naive Bayes Model

#install.packages("naivebayes")
library(naivebayes)

nb <- naive_bayes(target ~ ., data=data_train)

predicted <- predict(nb,data_test,type="class")

# Evaluate Model performance
install.packages("e1071")
library(e1071)
library(caret)
confusionMatrix(predicted,
                data_test$target,
                mode="everything")

# Overall Accuracy = 58.48%


# K-nn Model
library(class)
predicted <- knn(train = data_train[,-94], test = data_test[,-94],cl = data_train[,94], k=11)

# Evaluate Model performance
install.packages("e1071")
library(e1071)
library(caret)
confusionMatrix(predicted,
                data_test$target,
                mode="everything")

# Overall Accuracy = 76.07%


# CART Model
## install.packages("rpart")
## install.packages("rpart.plot")
library(rpart)
library(rpart.plot)

## setting the control paramter inputs for rpart
r.ctrl = rpart.control(minsplit=400, minbucket = 100, cp = 0, xval = 5)

m1 <- rpart(formula = target ~ ., data = data_train, method = "class", control = r.ctrl)

printcp(m1)

predicted <- predict(m1, data_test, type="class")

# Evaluate Model performance
install.packages("e1071")
library(e1071)
library(caret)
confusionMatrix(predicted,
                data_test$target,
                mode="everything")

# Overall Accuracy = 68.87%

# Random Forest
##install.packages("randomForest")
library(randomForest)


RF <- randomForest(target ~ ., data = data_train, 
                   ntree=500, mtry = 45, nodesize = 100,
                   importance=TRUE)

plot(RF)

predicted <- predict(RF, data_test, type="class")

# Evaluate Model performance
install.packages("e1071")
library(e1071)
library(caret)
confusionMatrix(predicted,
                data_test$target,
                mode="everything")

# Overall Accuracy = 77.31%

## Tuning Random Forest - To improve the above Model Accuracy

tuneRF <- tuneRF(x = data_train[,-94], 
                 y=data_train[,94],
                 mtryStart = 45, 
                 ntreeTry=81, 
                 stepFactor = 1, 
                 improve = 0.01, 
                 trace=TRUE, 
                 plot = TRUE,
                 doBest = TRUE,
                 nodesize = 100, 
                 importance=FALSE
)