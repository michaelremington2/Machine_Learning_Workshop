# https://www.kaggle.com/jsphyg/weather-dataset-rattle-package/code
# Variable metadata: http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml
library(dplyr)
library(ggplot2)
library(lubridate)
library(tidyr)
library(rpart)
library(randomForest)
library(caret)
library(pROC)
library(mice)

#RainTomorrow is the target variable to predict. It means -- did it rain the next day, Yes or No?
# This column is Yes if the rain for that day was 1mm or more.

# Load Data Set
rain<-read.csv("weatherAUS.csv")

#Lets take a look
View(rain)
str(rain)

# Look at all the unique Values in the data
lapply(rain, unique)

# ugh oh, lots of NAs, Lets check this out
count_na <- function(vector){
  return(sum(is.na(vector)))
    }
na_df<-t(data.frame(c(lapply(rain, count_na))))


# Percent of missing data in each column

pMiss <- function(x){sum(is.na(x))/length(x)*100}
apply(rain,2,pMiss)
apply(rain,1,pMiss)

# Lets remove the columns with lots of missing data
# Data with over 25% missing should be not be imputed
# Important to recognize random missing data vs systematic missing data

rain_df <- rain %>% select(-Sunshine,-Evaporation,-Cloud3pm,-Cloud9am)

## Imputed Data

impute_data <-  mice(rain_df,m=5,maxit=50,meth='pmm',seed=500) # predictive mean matching 


# Better ways to account for NAs are bootstrapping or imputing data with machine learning.
# Unfortunately I want to focus on other things as bootstrapping is complicated 
# and imputing lots of data can be computationally expensive
# A cool function I found for imputing

data.imputed <- rfImpute(RaintTomorrow~., data = rain, iter=6) 

# This was too computationally expensive for my poor little laptop though :(




rain_df$RainToday <- as.factor(rain_df$RainToday)
rain_df$RainTomorrow <- as.factor(rain_df$RainTomorrow)
rain_df$Location <- as.factor(rain_df$Location)


# Lets get the year, month, and day from the date using the lubridate package
rain_df <- rain_df %>%
  mutate(day = day(Date),
         yday=yday(Date),
         month = month(Date), 
         year = year(Date))

# For now, lets drop all rows with NA 
rain_df_1 <- rain_df %>% na.omit()


# Partition Data into Training and test sets
# Set seed so results are reproducable
set.seed(123)

#Partition data into training and test sets
row <- nrow(rain_df_1)
coll <- ncol(rain_df_1)
numTrain <- floor((2/3) * row)
numTest <- row - numTrain
training <- rain_df_1[sample(row, numTrain), ]
test <- rain_df_1[sample(row, numTest), ]


####### Model Building

help(randomForest)

fit <- randomForest(RainTomorrow~month+
                      yday+
                      Location+
                      MinTemp+
                      MaxTemp+ 
                      Rainfall+
                      RainToday,
                    data = training,
                    keep.forest=TRUE)

fit


##### Model Tuning OOB
# Error rate
oob.error.data <- data.frame(
  Trees=rep(1:nrow(fit$err.rate), times=1),
  Type=rep(c("OOB",
             "Yes",
             "No"), each=nrow(fit$err.rate)),
  Error=c(fit$err.rate[,"OOB"],
          fit$err.rate[,"Yes"],
          fit$err.rate[,"No"]))

error_rate<-ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type),size=3)+
  ggtitle('Out Of Bag Error Analysis')

error_rate

##################################
#### Accuracy and Recall #########
##################################

#### Predict the test set
forest.pred <- predict(fit, test)
con <- cbind(test,forest.pred)


# Accuracy, Recall
help("confusionMatrix")
result <- confusionMatrix(con$forest.pred, con$RainTomorrow, mode="prec_recall",positive = "Yes")

result
result$overall["Accuracy"]
result$byClass["Precision"]
result$byClass["Recall"] # Lots of false negatives
result$byClass["F1"]

#################################
#### ROC and AUC Curves #########
#################################

library(ROCR)

predictions <- predict(fit, test, type = "prob")[,2]
pred <- prediction(predictions, test$RainTomorrow)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, col=rainbow(10))


auc_ROCR <- performance(pred, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]
auc_ROCR


#cutoffs <- data.frame(cut=perf@alpha.values[[1]], fpr=perf@x.values[[1]], 
#                      tpr=perf@y.values[[1]])
#temp <- subset(cutoffs, fpr < 0.2 & tpr>0.9)

##################################
#
# Variable Importance
#
##################################

gini_score<- as.data.frame(importance(fit))
gini_score <- data.frame(
  names=c("month",
          "yday",
          "Location",
          "MinTemp",
          "MaxTemp",
          "Rainfall",
          "RainToday"),
  gini=importance(fit))


impg<-ggplot(data=gini_score, aes(x=reorder(names, -MeanDecreaseGini), y=MeanDecreaseGini)) +
  geom_bar(stat="identity", fill="steelblue")+
  geom_text(aes(label=round(MeanDecreaseGini,2)), vjust=1.6, color="white", size=3.5)+
  theme_minimal()+
  xlab('Variable')+
  ylab('Mean Decreasing Gini')+
  ggtitle('Variable Importance Measure Model #2')+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  xlab("Variable")

impg

################################################################
###########################         ############################
########################### Model 2 ############################
###########################         ############################
################################################################


fit2 <- randomForest(RainTomorrow~month+
                      yday+
                      Location+
                      MinTemp+
                      MaxTemp+ 
                      Rainfall+
                      RainToday+
                      Pressure3pm+
                      Pressure9am+
                      WindGustSpeed,
                    data = training,
                    mtry=3,
                    ntree=100,
                    keep.forest=TRUE)

fit2
gini_score<- as.data.frame(importance(fit2))

##### Model Tuning OOB
# Error rate
oob.error.data <- data.frame(
  Trees=rep(1:nrow(fit2$err.rate), times=1),
  Type=rep(c("OOB",
             "Yes",
             "No"), each=nrow(fit2$err.rate)),
  Error=c(fit2$err.rate[,"OOB"],
          fit2$err.rate[,"Yes"],
          fit2$err.rate[,"No"]))

error_rate<-ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type),size=3)+
  ggtitle('Out Of Bag Error Analysis')

error_rate

##################################
#### Accuracy and Recall #########
##################################

#### Predict the test set
forest.pred <- predict(fit2, test)
con <- cbind(test,forest.pred)


# Accuracy, Recall

result <- confusionMatrix(con$forest.pred, con$RainTomorrow, mode="prec_recall",positive = "Yes")

result
result$overall["Accuracy"]
result$byClass["Precision"]
result$byClass["Recall"]
result$byClass["F1"]

##################################
#
# Variable Importance
#
##################################

importance(fit2)
gini_score <- data.frame(
  names=c("month",
          "yday",
          "Location",
          "MinTemp",
          "MaxTemp",
          "Rainfall",
          "RainToday",
          "Pressure3pm",
          "Pressure9pm",
          "WindGustSpeed"),
  gini=importance(fit2))


impg<-ggplot(data=gini_score, aes(x=reorder(names, -MeanDecreaseGini), y=MeanDecreaseGini)) +
  geom_bar(stat="identity", fill="steelblue")+
  geom_text(aes(label=round(MeanDecreaseGini,2)), vjust=1.6, color="white", size=3.5)+
  theme_minimal()+
  xlab('Variable')+
  ylab('Mean Decreasing Gini')+
  ggtitle('Variable Importance Measure Model #2')+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  xlab("Variable")

impg

#################################
#### ROC and AUC Curves #########
#################################


predictions_2 <- predict(fit2, test, type = "prob")[,2]
pred_2 <- prediction(predictions_2, test$RainTomorrow)
perf_2 <- performance(pred_2, measure = "tpr", x.measure = "fpr")
plot(perf, col=rainbow(10))
plot(perf_2,col="green")


auc_ROCR <- performance(pred_2, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]
auc_ROCR






###################################
### Neural Networks
###################################



# library(neuralnet)
# help("neuralnet")

# nn_rain_df <- rain_df_1 %>% 
#   select(RainTomorrow,month,yday,MinTemp,MaxTemp,Location,Rainfall,RainToday,WindGustSpeed)
# 
# nn_rain_df[sapply(nn_rain_df, is.factor)] <- data.matrix(nn_rain_df[sapply(nn_rain_df, is.factor)])
# nn_rain_df <- nn_rain_df%>% mutate(RainTomorrow=ifelse(RainTomorrow==2, 1,0),
#                                                 RainToday=ifelse(RainToday==2, 1,0),)
# 
# 
# row <- nrow(nn_rain_df)
# coll <- ncol(nn_rain_df)
# numTrain <- floor((2/3) * row)
# numTest <- row - numTrain
# training <- nn_rain_df[sample(row, numTrain), ]
# test <- nn_rain_df[sample(row, numTest), ]
# nn <- neuralnet(RainTomorrow~month+yday+Location+MinTemp+MaxTemp+ Rainfall+RainToday+WindGustSpeed,data = training, hidden=1, linear.output=FALSE, threshold=0.01)
# nn$result.matrix
# plot(nn)
