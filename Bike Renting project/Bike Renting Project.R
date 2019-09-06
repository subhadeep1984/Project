rm(list=ls())
#Set working directory
setwd("/Edwisor/R")

# Loading the train and test data
bike_data=read.csv("/Users/subhadeep/Downloads/day.csv")
# The data has 731 observations and has 16 varibles.

library(lubridate)
library(dplyr)

glimpse(bike_data)

# Missing Value Analysis:-
missing_values = sapply(bike_data, function(x){sum(is.na(x))})
# There are no missing values in the data

## We are dropping 'casual' and 'registered' variable as the summation of these
# 2 varibles are given in 'cnt' column. 
bike_data_new= subset(bike_data,select = -c(casual,registered))

# parsing the dteday and taking only the day as new column as year and month
# are already there in the dataset.

bike_data_new$date=parse_date_time(bike_data_new$dteday,"ymd")
bike_data_new$day=day(bike_data_new$date)

# dropping variables instant,dteday,date
bike_data_new=subset(bike_data_new,select=-c(instant,dteday,date))

# further processing purpose we are keeping the origial data in df
df=bike_data_new

# we are adding 4 new columns in df data for further insights
df$actual_season = factor(x = df$season, levels = c(1,2,3,4), labels = c("Spring","Summer","Fall","Winter"))
df$actual_yr = factor(x = df$yr, levels = c(0,1), labels = c("2011","2012"))
df$actual_holiday = factor(x = df$holiday, levels = c(0,1), labels = c("Working day","Holiday"))
df$actual_weathersit = factor(x = df$weathersit, levels = c(1,2,3,4), 
                               labels = c("Clear","Cloudy/Mist","Rain/Snow/Fog","Heavy Rain/Snow/Fog"))


sort(xtabs(formula = cnt~actual_season,df))

#actual_season
#Spring  Winter  Summer    Fall 
#471348  841613  918589 1061129 

# here we can see that bike was rented highest in Fall and lowest in spring.

xtabs(formula = cnt~actual_holiday,df)

#actual_holiday
#Working day     Holiday 
#3214244       78435 
# here we can see that bike was rented heavily during working days

xtabs(formula = cnt~actual_weathersit,df)
#actual_weathersit
#Clear         Cloudy/Mist       Rain/Snow/Fog         Heavy Rain/Snow/Fog 
#2257952              996858               37869                   0 

# clearly we can see that bike was rented highest during clear weather and zero
# during heavy rain/snow

xtabs(formula = cnt~actual_yr,df)

#actual_yr
#2011    2012 
#1243103 2049576 

# bike was rented more in 2012 year.
#####################################

# OUTLIER Analysis:-

# making a new dataframe with numeric variables only
library(tidyverse)
cnames = colnames(bike_data_new[,c("temp","atemp","hum","windspeed")])
for(i in cnames){
  val = bike_data_new[,i][bike_data_new %in% boxplot.stats(bike_data_new[,i])$out]
  print(length(val))
}

boxplot.stats(bike_data_new$temp)
# there are no outliers present in temp variable

boxplot.stats(bike_data_new$windspeed)
#$out
#[1] 0.417908 0.507463 0.385571 0.388067 0.422275 0.415429 0.409212 0.421642 0.441563
#[10] 0.414800 0.386821 0.398008 0.407346

# max whiskers point is 0.37 but max point in the windspeed column is 0.50746.
# so we are not deleting or replacing with any values keepig i mind that this are 
# not ouliers as there might some windy situations for that the windspeed has
# incresed to some extent

boxplot.stats(bike_data_new$hum)
# there are 2 ouliers but we are kepping it same .

boxplot.stats(bike_data_new$atemp)
#No outlier present in the data.
#############################
#Feature scaling :-

# Correlation plot for numeric variables:-
library(corrgram)
library(usdm)
num_names=bike_data_new[,c("temp","atemp","hum","windspeed","day")]

vifcor(num_names)

corrgram(num_names, order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

# we can see that atemp is highly correlated to temp. so we are removing that variable.
bike_data_new=subset(bike_data_new,select=-c(atemp))
##########################################################


# Normality check
hist(bike_data_new$temp)
hist(bike_data_new$hum)
hist(bike_data_new$windspeed)

# We can see that the data is not normally distributed, so we are normalizing the data
cnames1 = colnames(bike_data_new[,c("temp","hum","windspeed")])
for ( i in cnames1) {
  print(i)
  bike_data_new[,i]=(bike_data_new[,i]-min(bike_data_new[,i]))/
    (max(bike_data_new[,i]-min(bike_data_new[,i])))
} 

######################################################

## MODEL DEVELOPMENT :-

set.seed(2)
s=sample(1:nrow(bike_data_new),0.75*nrow(bike_data_new))
train=bike_data_new[s,]
test=bike_data_new[-s,]

fit=lm(cnt~.,data=train)
library(car)
vif(fit)
# all the vif scores are less than 5 which is acceptable.

step(fit)

summary(fit) 
# on the base of p value we can remove varibles which has pvalue>0.5
# we can drop variables like workingday,day.
# though we can remove variable hum based o p  value but as its p value is in border
# line we are accepting for the final model

fit_final=lm(cnt~season+yr+mnth+holiday+weekday+weathersit+temp+windspeed+hum,data=train)
summary(fit_final)

pred=predict(fit_final,newdata = test)

# Calculating the RMSE 
library(caret)
RMSE(test$cnt,pred)
# rmse is 812.7414

# other way for error mterices
library(DMwR)
regr.eval(test$cnt,pred,stats = c('rmse','mape'))
#    rmse       mape 
# 812.7414   0.1904365 
# so the above linear regression is accurate of 80.96% while error rate is 19.04%
# Visual 

# 1. Residual vs. Fitted Values Plot:-
plot(fit_final,which = 1)

# 2. Normality Q-Q Plot:-
plot(fit_final,which = 2)

# 3. Scale Location Plot:-
plot(fit_final,which = 3)

# 4. Outliers detection : None found , cook's distance < 1 for all obs
plot(fit_final,which = 4)


#######################################

## RANDOM FOREST :-

# train model
control =trainControl(method="repeatedcv", number=10, repeats=3,search = 'random')

set.seed(2)
rf =train(cnt~., data=train, method='rf',tuneLength=20 ,trControl=control)

plot(rf)

print(rf)

# we are choosing mtry=6 as in thsi case RMSE seems to be the lowest.
library(randomForest)
rf_bike_rental=randomForest(cnt~.,train,importance=TRUE,ntree=100,mtry=6)
importance(rf_bike_rental)
print(rf_bike_rental)
varImpPlot(rf_bike_rental)
varImp(rf_bike_rental)
pred_rf=predict(rf_bike_rental,newdata = test)

regr.eval(test$cnt,pred_rf,stats = c('rmse','mape'))
# rmse        mape 
# 674.40841   0.1614294
# here we can see that the model has improved compare to linear regression.
# the model is 83.86 % accurate while error rate is 16.14%.





