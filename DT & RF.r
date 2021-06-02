####################### problem 1 ###############################
######################decison tree########################
# company_data.csv

company_data <- read.csv(file.choose())
View(company_data)
##Exploring and preparing the data 
str(company_data)

# look at the class variable
table(company_data$US)

#categorical variables need to be set as factor variables.
company_data$US <- as.factor(company_data$US)
company_data$Sales <- as.factor(company_data$Sales)

#converting numerical into categorical data.
#cloth$Category[cloth$Sales == 1 | cloth$Sales == 2] = "Low"
#cloth$Category[cloth$Sales == 3                   ] = "Medium"
#cloth$Category[cloth$sales == 4 | cloth$sales == 5] = "High"
#cloth

# Shuffle the data
company_data_rand <- company_data[order(runif(400)), ]
str(company_data_rand)

# split the data frames
company_data_train <- company_data_rand[1:300, ]
company_data_test  <- company_data_rand[301:400, ]

# check the proportion of class variable
prop.table(table(company_data_rand$US))
prop.table(table(company_data_train$US))
prop.table(table(company_data_test$US))

# Step 3: Training a model on the data
install.packages("C50")
library(C50)

# C5.0 models require a factor outcome. You have given the outcome as credit_train$default, which is a 1/2 outcome, but R has read it as numeric, rather than a factor:
company_data_train$US<-as.factor(company_data_train$US)
str(company_data_train$US)
company_data_model <- C5.0(company_data_train[-12], company_data_train$US)

windows()
plot(company_data_model) 

# Display detailed information about the tree
summary(company_data_model)

# Step 4: Evaluating model performance
# Test data accuracy
test_res <- predict(company_data_model, company_data_test)
test_acc <- mean(company_data_test$US == test_res)
test_acc   #1

# cross tabulation of predicted vs actual classes
library(gmodels)
CrossTable(company_data_test$US, test_res, dnn = c('actual US', 'predicted US'))

# On Training Dataset
train_res <- predict(company_data_model, company_data_train)
train_acc <- mean(company_data_train$US == train_res)
train_acc  #1

table(company_data_train$US, train_res)

######################  RANDOM FOREST    ######################## 
#Underfit , over, Right
# Load the Data
# company.csv
company <- read.csv(file.choose())

##Exploring and preparing the data
str(company)

library(caTools)
set.seed(0)
split <- sample.split(company$Income, SplitRatio = 0.8)
company_train <- subset(company, split == TRUE)
company_test <- subset(company, split == FALSE)

#model building
install.packages("randomForest")
library(randomForest)
#Type rfNews() to see new features/changes/bug fixes.
rfNews()
rf <- randomForest(company_train$Income ~ ., data = company_train)
# Default 'mtry' value will be equal p/3
# 17/3 = 5.66 = 6 (rounded)

# Prediction for test data result
test_rf_pred <- predict(rf, company_test)
# RMSE on Test Data
rmse_rf <- sqrt(mean(company_test$Income - test_rf_pred)^2)
rmse_rf

# Prediction for trained data result
train_rf_pred <- predict(rf, company_train)

# RMSE on Train Data
train_rmse_rf <- sqrt(mean(company_train$Income - train_rf_pred)^2)
train_rmse_rf

#############################Probelm 2########################################

# Diabetes.csv

diabetes <- read.csv(file.choose())
View(diabetes)
##Exploring and preparing the data
str(diabetes)

# look at the class variable
table(diabetes$Class.variable)

diabetes$Class.variable <- as.factor(diabetes$Class.variable)
diabetes$Number.of.times.pregnant <- as.factor(diabetes$Number.of.times.pregnant)

# Shuffle the data
diabetes_rand <- diabetes[order(runif(768)), ]
str(diabetes_rand)

# split the data frames
diabetes_train <- diabetes_rand[1:600, ]
diabetes_test  <- diabetes_rand[601:768, ]

# check the proportion of class variable
prop.table(table(diabetes$Class.variable))
prop.table(table(diabetes_train$Class.variable))
prop.table(table(diabetes_test$Class.variable))

# Step 3: Training a model on the data
install.packages("C50")
library(C50)

diabetes_model <- C5.0(diabetes_train[], diabetes_train$Class.variable)

windows()
plot(diabetes_model) 

# Display detailed information about the tree
summary(diabetes_model)

# Step 4: Evaluating model performance
# Test data accuracy
test_res <- predict(diabetes_model, diabetes_test)
test_acc <- mean(diabetes_test$Class.variable == test_res)
test_acc   #1

# cross tabulation of predicted vs actual classes
library(gmodels)
CrossTable(diabetes_test$Class.variable, test_res, dnn = c('actual Class.variable', 'predicted Class.variable'))

# On Training Dataset
train_res <- predict(diabetes_model, diabetes_train)
train_acc <- mean(diabetes_train$Class.variable == train_res)
train_acc  #1

table(diabetes_train$Class.variable, train_res)

######################  RANDOM FOREST    ########################
# Load the Data
# diabetes1.csv
diabetes1 <- read.csv(file.choose())

##Exploring and preparing the data ----
str(diabetes1)

library(caTools)
set.seed(0)
split <- sample.split(diabetes1$Class.variable, SplitRatio = 0.8)
diabetes1_train <- subset(diabetes1, split == TRUE)
diabetes1_test <- subset(diabetes1, split == FALSE)


# install.packages("randomForest")
library(randomForest)

rf <- randomForest(diabetes1_train$Number.of.times.pregnant ~ ., data = diabetes1_train)
# Default 'mtry' value will be equal p/3
# 17/3 = 5.66 = 6 (rounded)
# Prediction for test data result
test_rf_pred <- predict(rf, diabetes1_test)
# RMSE on Test Data
rmse_rf <- sqrt(mean(diabetes1_test$Number.of.times.pregnant - test_rf_pred)^2)
rmse_rf

# Prediction for trained data result
train_rf_pred <- predict(rf, diabetes1_train)

# RMSE on Train Data
train_rmse_rf <- sqrt(mean(diabetes1_train$Number.of.times.pregnant - train_rf_pred)^2)
train_rmse_rf

###################### problem 3 #######################
#decison tree####
## Load the Data
fcheck.csv

fcheck <- read.csv(file.choose())
##Exploring and preparing the data 
str(fcheck)

# look at the class variable
table(fcheck$Taxable.Income)

fcheck$Taxable.Income <- as.factor(fcheck$Taxable.Income)
#credit$checking_balance <- as.factor(credit$checking_balance)

# Shuffle the data
fcheck_rand <- fcheck[order(runif(600)), ]
str(fcheck_rand)

# split the data frames
fcheck_train <- fcheck_rand[1:500, ]
fcheck_test  <- fcheck_rand[501:600, ]

# check the proportion of class variable
prop.table(table(fcheck_rand$Taxable.Income))
prop.table(table(fcheck_train$Taxable.Income))
prop.table(table(fcheck_test$Taxable.Income))

# Step 3: Training a model on the data
install.packages("C50")
library(C50)

fcheck_model <- C5.0(fcheck_train[], fcheck_train$Taxable.Income)

windows()
plot(fcheck_model) 

# Display detailed information about the tree
summary(fcheck_model)

# Step 4: Evaluating model performance
# Test data accuracy
test_res <- predict(fcheck_model, fcheck_test)
test_acc <- mean(fcheck_test$Taxable.Income == test_res)
test_acc #0

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(fcheck_test$Taxable.Income, test_res, dnn = c('actual Taxable.Income', 'predicted Taxable.Income'))

# On Training Dataset
train_res <- predict(fcheck_model, fcheck_train)
train_acc <- mean(fcheck_train$Taxable.Income == train_res)
train_acc #0.432

table(fcheck_train$Taxable.Income, train_res)
###################   RANDOM FOREST   ##################################
# Load the Data
# fcheck1.csv
fcheck1 <- read.csv(file.choose())

##Exploring and preparing the data ----
str(fcheck1)

library(caTools)
set.seed(0)
split <- sample.split(fcheck1$Taxable.Income, SplitRatio = 0.8)
fcheck1_train <- subset(fcheck1, split == TRUE)
fcheck1_test <- subset(fcheck1, split == FALSE)


# install.packages("randomForest")
library(randomForest)

rf <- randomForest(fcheck1_train$Taxable.Income ~ ., data = fcheck1_train)
# Default 'mtry' value will be equal p/3
# 17/3 = 5.66 = 6 (rounded)

test_rf_pred <- predict(rf, fcheck1_test)

rmse_rf <- sqrt(mean(fcheck1_test$Taxable.Income - test_rf_pred)^2)
rmse_rf  #1643.291

# Prediction for trained data result
train_rf_pred <- predict(rf, fcheck1_train)

# RMSE on Train Data
train_rmse_rf <- sqrt(mean(fcheck1_train$Taxable.Income - train_rf_pred)^2)
train_rmse_rf  #102.3139

#############################problem 4 #####################
#####decision tree#########
# HR_DT.csv

HR_DT <- read.csv(file.choose())
##Exploring and preparing the data ----
str(HR_DT)

# look at the class variable
table(HR_DT$Position.of.the.employee)
table(HR_DT$monthly.income.of.employee)

HR_DT$Position.of.the.employee <- as.factor(HR_DT$Position.of.the.employee)
HR_DT$monthly.income.of.employee <- as.factor(HR_DT$monthly.income.of.employee)


#HR_DT$RegionManager <- as.factor(HR_DT$RegionManager)

# Shuffle the data
HR_DT_rand <- HR_DT[order(runif(196)), ]
str(HR_DT_rand)

# split the data frames
HR_DT_train <- HR_DT_rand[1:150, ]
HR_DT_test  <- HR_DT_rand[151:196, ]

# check the proportion of class variable
prop.table(table(HR_DT_rand$Position.of.the.employee))
prop.table(table(HR_DT_train$Position.of.the.employee))
prop.table(table(HR_DT_test$Position.of.the.employee))

# Step 3: Training a model on the data
install.packages("C50")
library(C50)

#HR_DT_model <- C5.0(HR_DT_train[], HR_DT_train$Position.of.the.employee)
HR_DT_model <- C5.0(HR_DT_train[], HR_DT_train$monthly.income.of.employee)

windows()
plot(HR_DT_model) 

# Display detailed information about the tree
summary(HR_DT_model)

# Step 4: Evaluating model performance
# Test data accuracy
test_res <- predict(HR_DT_model, HR_DT_test)
test_acc <- mean(HR_DT_test$monthly.income.of.employee == test_res)
test_acc #1

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(HR_DT_test$monthly.income.of.employee, test_res, dnn = c('actual monthly.income.of.employee', 'predicted monthly.income.of.employee'))

# On Training Dataset
train_res <- predict(HR_DT_model, HR_DT_train)
train_acc <- mean(HR_DT_train$Position.of.the.employee == train_res)
train_acc  #0

table(HR_DT_train$Position.of.the.employee, train_res)
###################   RANDOM FOREST   ##############################
# Load the Data
# HR_DT.csv
HR <- read.csv(file.choose())

##Exploring and preparing the data
str(HR)

library(caTools)
set.seed(0)
split <- sample.split(HR$Position.of.the.employee, SplitRatio = 0.8)
HR_train <- subset(HR, split == TRUE)
HR_test <- subset(HR, split == FALSE)


# install.packages("randomForest")
library(randomForest)

rf <- randomForest(HR_train$monthly.income.of.employee ~ ., data = HR_train)
# Default 'mtry' value will be equal p/3
# 17/3 = 5.66 = 6 (rounded)
#Prediction for test data result
test_rf_pred <- predict(rf, HR_test)
# RMSE on Train Data
rmse_rf <- sqrt(mean(HR_test$monthly.income.of.employee - test_rf_pred)^2)
rmse_rf #181.052

# Prediction for trained data result
train_rf_pred <- predict(rf, HR_train)

# RMSE on Train Data
train_rmse_rf <- sqrt(mean(HR_train$monthly.income.of.employee - train_rf_pred)^2)
train_rmse_rf  #141.1797


