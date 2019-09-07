rm(list=ls())

#Set working directory
setwd("/Edwisor/R")

# Loading the train and test data
bd_train=read.csv("/Users/subhadeep/Downloads/train (1).csv")
bd_test=read.csv("/Users/subhadeep/Downloads/test (1).csv")

# Dimesions of the data set
dim(bd_train)
dim(bd_test)

# Summary of the training and test data
str(bd_train)
summary(bd_train)

## Dropping 'target' and ' ID_code' from train and 'ID_code' from test data
bd_train_new= subset(bd_train, select = -c(target,ID_code))
bd_test_new=subset(bd_test,select= -c(ID_code))

# checking for target class variable
table(bd_train$target)
#Percenatge counts of target classes
table(bd_train$target)/length(bd_train$target)*100
#Bar plot for count of target classes
library(ggplot2)
ggplot(bd_train,aes(target))+theme_bw()+geom_bar(stat='count',fill='lightgreen')
## This is imbalanced dataset ( ratio almost 90:10)
# where 90% of the data is the data of number of customers those will not make a transaction
#and 10% of the data is those who will make a transaction.

# # Let us look into distribution of the Training Data and Test Data
library(tidyverse)
hist_data=bd_train_new %>% gather() %>% head()

ggplot(gather(bd_train_new[,c(1:50)]), aes(value)) + 
  geom_histogram(bins = 10) + 
  facet_wrap(~key, scales = 'free_x')

ggplot(gather(bd_train_new[,c(51:100)]), aes(value)) + 
  geom_histogram(bins = 10) + 
  facet_wrap(~key, scales = 'free_x')

ggplot(gather(bd_train_new[,c(101:150)]), aes(value)) + 
  geom_histogram(bins = 10) + 
  facet_wrap(~key, scales = 'free_x')

ggplot(gather(bd_train_new[,c(151:200)]), aes(value)) + 
  geom_histogram(bins = 10) + 
  facet_wrap(~key, scales = 'free_x')

## Test data
hist_data_test=bd_test_new %>% gather() %>% head()

ggplot(gather(bd_test_new[,c(1:50)]), aes(value)) + 
  geom_histogram(bins = 10) + 
  facet_wrap(~key, scales = 'free_x')

ggplot(gather(bd_test_new[,c(51:100)]), aes(value)) + 
  geom_histogram(bins = 10) + 
  facet_wrap(~key, scales = 'free_x')

ggplot(gather(bd_test_new[,c(101:150)]), aes(value)) + 
  geom_histogram(bins = 10) + 
  facet_wrap(~key, scales = 'free_x')

ggplot(gather(bd_test_new[,c(151:200)]), aes(value)) + 
  geom_histogram(bins = 10) + 
  facet_wrap(~key, scales = 'free_x')


## Misising Value Analysis fro train and test data
missing_val= data.frame(apply(bd_train_new, 2,function(x){sum(is.na(x))}))
missing_test=data.frame(apply(bd_test_new,2,function(x){sum(is.na(x))}))

# So there are no missing value present in train and test data

## OUTLIER Analysis:
boxplot(bd_train_new$var_0)
boxplot(bd_train_new$var_3)

b=boxplot(bd_train_new$var_0)
b

# when we analyse the outliers present in var_0 variable , we can see that there some 
# outliers present in the data but not to the maximum limit. 
# like if we see in var_0 variable in boxplot the whiskers range is 
## 2.004 and 19.2135  , where as the minimum and maximum point in var_0 is 
# 0.4084 and 20.315. So from here we can make 2 data set consists of 
# outlier analysis and without outlier annalysis.
# if we look all the variable we can see that all the outliers are similar to 
# var_0 outlier position.


df_new=bd_train_new
df=bd_train
# here we are copying the 2 dataset in df_new and df so that we can utilize
# the original data later if required
cnames= colnames(bd_train_new %>% select(1:200))
for(i in cnames){
  val = bd_train_new[,i][bd_train_new[,i] %in% boxplot.stats(bd_train_new[,i])$out]
  print(length(val))
  bd_train_new[,i][bd_train_new[,i] %in% val] = NA
}
# Above line: we have identified the outlier data in the data and replaced
# with NAN values.

missing_val_outlier= data.frame(apply(bd_train_new, 2,function(x){sum(is.na(x))}))
# now if we check there are NAN values in each column
for (i in cnames){
  bd_train_new[,i][is.na(bd_train_new[,i])]=
    mean(bd_train_new[,i], na.rm = T)
}
missing_value_check=data.frame(apply(bd_train_new, 2,function(x){sum(is.na(x))}))

# we are cheking again if there are any nan values after imputing with mean value 
# in place of NAN values.

## Correlation check

library(corrplot)
cor_plot=cor(bd_train_new)
head(round(cor_plot,2))
corrplot(cor_plot,method = "color")

## If you see there are no such correlation between the independent variables

## NORMALITY Check:
qqnorm(bd_train_new$var_0)
hist(bd_train_new$var_0)

qqnorm(bd_train_new$var_33)
hist(bd_train_new$var_33)

qqnorm(bd_train_new$var_44)
hist(bd_train_new$var_44)

# In the Normality ckeck we have take randomly some histogram of different variables
# not all the variables have normaly distributed like var_33 or var_44
# So we go for Normalization

for ( i in cnames) {
  print(i)
  bd_train_new[,i]=(bd_train_new[,i]-min(bd_train_new[,i]))/
    (max(bd_train_new[,i]-min(bd_train_new[,i])))
} 

## MODEL DEVELOPMENT:

# adding the target varible to bd_train_new datafram
bd_train_new$target=bd_train$target

library(DMwR)
library(randomForest)
library(caret)

## stratified sampling 
set.seed(1234)
train.index=createDataPartition(bd_train_new$target, p=.8, list= FALSE)
train1=bd_train_new[train.index,]
test1=bd_train_new[-train.index,]

prop.table(table(train1$target))
# the Data is imbalanced . only 10% are positives

## LOGISTIC REGRESSION:
model.logit=glm(target~.,data = train1,family = 'binomial')

# Summary of the model
summary(model.logit)
## We have got maximum and minimum error as 3.77 and -2.635
# so there are not much variation in the error 
## variables like var_0, var_1,var_6 can explin the variance of target varible much
# category like those who have p value> 0.5 like var_7,var_17,var_27 can't explai much 
# about target variable

# null devaince explains how well the target varible can be explained by the model
# with only coefficients.
# Residual deviance explains while including all the variables.
# null deviance and residula deviance is 104505 and 74373 respectively
# Lower value of residual deviance points out that the model has become better 
# when it has included 200 variables
# The degrees of freedom for null deviance equals N−1, where N is the number
# of observations in data sample.Here N=160000,therefore N-1=160000-1=159999
# The degrees of freedom for residual deviance equals N−k−1, where k is the 
#number of variables and N is the number of observations in data sample.
# Here N=160000,k=200 ,therefore N-k-1=160000-200-1=159799


# predict using logistic regression
logit_prediction=predict(model.logit,newdata = test1,type = "response")

# converting into probabilities
logit_prediction=ifelse(logit_prediction > 0.5, 1, 0)

# confusing matrix:
confmatrix_log=table(logit_prediction,test1$target)
#  logit_prediction    
#         0      1
#   0   35530  2872
#  1   481    1117

# OR

conf_log=confusionMatrix(as.factor(logit_prediction),as.factor(test1$target))

#             Reference
# Prediction     0       1
#     0       35530   2872
#     1        481    1117

# True positive (TP)= 1117
# True Negative (TN)= 35530
# False positive (FP)= 481
# False Negative (FN)= 2872

# Accuracy= (TP+TN)/(TP+TN+FP+FN)= 36644/39997 =0.9162
# Precision= TP/(TP+FP)=1117/(1117+481) =0.699
# Recall= TP/(TP+FN)= 1117/(1117+2872) =0.28
# FNR= FN/(FN+TP)=2872/3949 = 0.72

## variable importance plot
varImp(model.logit)

library(pROC)
library(ROCR)

# we are measuring the auc of th emodel
auc(test1$target,logit_prediction)

# so area under the curve is 0.633

## RANDOMFOREST:
RF_model = randomForest(target ~ ., data=train1, importance = TRUE, ntree = 50)

#Presdict test data using random forest model
RF_Predictions = predict(RF_model, test1[,-201])

# AUC score of the model
test1$target=as.numeric(as.character(test1$target))
RF_Predictions=as.numeric(as.character(test1$target))

# we are converting both the factors into numeric
auc(test1$target,RF_Predictions)#
# auc score came as 0.5006.

##Evaluate the performance of classification model
conf_log_rf=confusionMatrix(as.factor(RF_Predictions),as.factor(test1$target))

#                 Reference
#Prediction       0       1
#            0   36011  3984
#            1     0     5

# True positive (TP)= 5
# True Negative (TN)= 36011
# False positive (FP)= 0
# False Negative (FN)= 3984

# Accuracy= (TP+TN)/(TP+TN+FP+FN)= 36644/39997 =0.9004
# Precision= TP/(TP+FP)=1117/(1117+481) = 1
# Recall= TP/(TP+FN)= 1117/(1117+2872) =0.0012
# FNR= FN/(FN+TP)=2872/3949 = 0.99

# here in Random forest auc score is 0.5 which is very low. The most important
# thing is that FNR and recall va;lue is 0.99 and 0.0012 respectively,
# whcih is very poor.

importance(RF_model)

varImpPlot(RF_model)



