# IBM HR Data Analysis- Employee Attrition

# Introduction: To predict if an employee is going to resign or not
# Objective:- More and more, we have to envestigate that, how the company objective factors influence in attrition of employees, and what kind of working enviroment most will cause employees attrition.

# Objective
# The company wants to understand what factors contributed most to employee Attrition and to create a model that 
# can predict if a certain employee will leave the company or not. The goal is to create or improve different 
# retention strategies on targeted employees. Overall, the implementation of this model will allow management to
# create better decision-making actions.
# Load the data
setwd("data")
HR_emp_att<-read.csv("pfm_train.csv",header = T,sep = ",",na.strings = c("",NA))

# Required library
library(ggplot2)
library(data.table)
library(highcharter)
library(caret)
library(VIM)
library(plyr)
library(dplyr)

library(caTools)
library(gridExtra)
library(DT)
library(randomForest)
library(Metrics)
library(pROC)
library(e1071)
library(dtree)
library(corrplot)
library(DMwR)
library(xgboost)
library(Hmisc)

# View the data
dim(HR_emp_att)
str(HR_emp_att)

## to check uniqueness of each variable
sapply(HR_emp_att, function(x) length(unique(x)))
# to check the class of each variable
t(sapply(HR_emp_att, class))
# to check summary of the Data
summary(HR_emp_att)
describe(HR_emp_att)

# As we see in the Data:

# Observations: 1,100 with Variables: 31

# Class Label is Attrition with 922 'NO' and 178 'Yes' that shows the unbalance class label. we have to pay attention to the unbalance class algorithm problems!

# Over 18 is equal to 'Y', which means employee is not less than 18 years old. this attribute should be considered for the future, maybe by changing the ruls of emploement, young people under 18 can also working in companies. Here, according to the data set, we will remove it.

# Moreover, Standard Hours is equal 80 for all observation. the decision for this attribute is same to Over18 and Employee Count.

# BusinessTravel, Department, EducationField, Gender, jobRole, MaritalStatus and OverTime are categorical data and other variabels are continues.

# Some of variables are related to the years of working wich can be a good candidate for feature generation.

# Some of variable are related to personal issues like WorkLifeBalance, RelationshipSatisfaction, JobSatisfaction,EnvironmentSatisfaction etc.

# There are some variables that are related to the income like MonthlyIncome, PercentSalaryHike, etc.

# EmployeeNumber is a variable for identifying the specific employee.If we have more information about employee and the structure of the employee number, then we can extract some new features. But now it is not possible and we have to remove it from our data set.

# Checking distribution of Attrition
HR_emp_att %>% group_by(Attrition) %>% summarise(cnt=n())

# Checking for missing values.
sapply(HR_emp_att, function(x) sum(is.na(x)))
# checking the variable importance
VIM::aggr(HR_emp_att)
# No Missing Value,we shall Remove non value attributes because These variables can not play significant role because they are same for all records.
# also, EmployeeNumber can be accepted as an indicator for the time of join to the company which can be used for new feature generation,But we do not have any meta data about it, then, we will remove it.

# Removing non value attributes
HR_emp_att$EmployeeNumber<- NULL
HR_emp_att$StandardHours <- NULL
HR_emp_att$Over18 <- NULL
HR_emp_att$EmployeeCount <- NULL
cat("Data Set has ",dim(HR_emp_att)[1], " Rows and ", dim(HR_emp_att)[2], " Columns" )

# There are some attributes that are categorical, but in the data set are integer. We have to change them to categorical.
HR_emp_att$Education <- factor(HR_emp_att$Education)
HR_emp_att$EnvironmentSatisfaction <- factor(HR_emp_att$EnvironmentSatisfaction)
HR_emp_att$JobInvolvement <- factor(HR_emp_att$JobInvolvement)
HR_emp_att$JobLevel <- factor(HR_emp_att$JobLevel)
HR_emp_att$JobSatisfaction <- factor(HR_emp_att$JobSatisfaction)
HR_emp_att$PerformanceRating <- factor(HR_emp_att$PerformanceRating)
HR_emp_att$RelationshipSatisfaction <- factor(HR_emp_att$RelationshipSatisfaction)
HR_emp_att$StockOptionLevel <- factor(HR_emp_att$StockOptionLevel)
HR_emp_att$WorkLifeBalance <- factor(HR_emp_att$WorkLifeBalance)
HR_emp_att$Attrition <- factor(HR_emp_att$Attrition)

## Exploratory Data Analysis

g1 <- ggplot(HR_emp_att, aes(x = MonthlyIncome, fill = Attrition)) + geom_density(alpha = 0.7) + scale_fill_manual(values = c("#386cb0","#fdb462")) 

### Attrition - log(Monthly Income)
g2<-ggplot(HR_emp_att, aes(x =  log(MonthlyIncome), fill = Attrition,colour = Attrition)) + geom_density( alpha = .3) + ggtitle("") 

grid.arrange(g1, g2, ncol = 1, nrow = 2)

# We can only make few observations. It seems fair to say that a large majority of
# those who left had a relatively lower monthly income and daily rate while the never
# made it in majority into the higher hourly rate group. On the other hand, the
# differences become elusive when comparing the monthly rate.

### YearsAtCompany - Attrition 
ggplot(HR_emp_att, aes(x = YearsAtCompany, fill = Attrition, colour = Attrition)) + geom_density( alpha = .3)

### Count - EducationField
attrition_edufield <- HR_emp_att %>%select(Attrition, EducationField) %>%group_by(Attrition, EducationField) %>% summarise(count = n()) + facet_wrap(~Attrition)
# creating datatable
datatable(attrition_edufield)

ggplot(attrition_edufield, aes(x = EducationField,y = count,fill = EducationField, colour = EducationField))+ geom_bar(stat = "identity")  + facet_wrap(~Attrition)

ggplot(HR_emp_att, aes(x = Age, fill = EducationField,colour = EducationField, alpha = .3)) + geom_density()

ggplot(HR_emp_att, aes(x =  Age, fill = BusinessTravel,colour = BusinessTravel, alpha = .3)) + geom_density()

### Distribution of Age/Overtime 
ggplot(HR_emp_att, aes(x =  Age, fill = OverTime, colour = OverTime, alpha = .3)) +geom_density()

### Distribution of Age
ggplot(data=HR_emp_att, aes(HR_emp_att$Age)) + geom_histogram(breaks=seq(20, 50, by=2),
                                                              col="red",aes(fill=..count..))+ labs(x="Age", y="Count")+ scale_fill_gradient("Count", low="black", high="red")  
# As we saw above, the majority of employees are between 28-36 years.

# Exploring BusinessTravel variable
a1 <- HR_emp_att %>%group_by(BusinessTravel) %>% tally() %>%ggplot(aes(x = BusinessTravel, y = n,fill=BusinessTravel)) +
  geom_bar(stat = "identity") +theme_minimal()+ labs(x="Business Travel", y="Number Attriation")+
  ggtitle("Attrition according to the Business Travel")+geom_text(aes(label = n), vjust = -0.5, position = position_dodge(0.9))

a2<- HR_emp_att %>% group_by(BusinessTravel, Attrition) %>%tally() %>%ggplot(aes(x = BusinessTravel, y = n,fill=Attrition)) +
  geom_bar(stat = "identity") + theme_minimal()+ labs(x="Business Travel", y="Number Attriation")+
  ggtitle("Attrition according to the Business Travel")+geom_text(aes(label = n), vjust = -0.5, position = position_dodge(0.8))

grid.arrange(a1,a2) 

# # Here is the distribution of the data according to the Business Tralvel situation.
# more than 70% of employees travel rarely and 10 % of them has no travel and 20% of them
# are travel frequently

HR_emp_att %>%ggplot(aes(x = BusinessTravel, group = Attrition)) +geom_bar(aes(y = ..prop.., fill = factor(..x..)),
                                                                           stat="count",alpha = 0.7) +geom_text(aes(label = scales::percent(..prop..), y = ..prop.. ),   
                                                                                                                stat= "count",vjust = 2) +labs(y = "Percentage", fill= "business Travel") + facet_grid(~Attrition) +   
  theme_minimal()+theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) + ggtitle("Attrition")           
# As it is obvious from the above plot that majority of employee those who Travel_Rarely had left
# Furthermore, there seems to be a clear indication that those who left travelled more frequently compared to
# others. This might have also been an important reason behind their resignation. Here I presume that travelling
# means staying somewhere else overnight, or for a longer period of time, which is why we may connect this to 
# work-life balance issues

# Exploring variable Department
g1 <- HR_emp_att %>%group_by(Department) %>%tally() %>%ggplot(aes(x = Department, y = n,fill=Department)) +
  geom_bar(stat = "identity") + theme_minimal()+geom_text(aes(label = n), vjust = -0.1, position = position_dodge(0.9)) 

g2 <- HR_emp_att %>%group_by(Department, Attrition) %>%tally() %>%ggplot(aes(x = Department, y = n,fill=Attrition)) +
  geom_bar(stat = "identity") +theme_minimal()+geom_text(aes(label = n), vjust = -0.1, position = position_dodge(0.9))  

grid.arrange(g1,g2) 


# Exploring the Variable Education
g1<- HR_emp_att %>%group_by(Education, Attrition) %>%tally() %>%ggplot(aes(x = Education, y = n,fill=Attrition)) +
  geom_bar(stat = "identity") + theme_minimal()+geom_text(aes(label = n), vjust = -0.5, position = position_dodge(0.9))+
  labs(x="Education", y="Number Attriation")+ggtitle("Attrition in regards to Education Level")

g2<- HR_emp_att %>%ggplot(aes(x = Education, group = Attrition)) +geom_bar(aes(y = ..prop.., fill = factor(..x..)),
                                                                           stat="count",alpha = 0.7) + geom_text(aes(label = scales::percent(..prop..), y = ..prop.. ),  
                                                                                                                 stat= "count",vjust = 2) +labs(y = "Percentage", fill= "Education") + facet_grid(~Attrition) +  
  theme_minimal()+ theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) + ggtitle("Attrition")           

grid.arrange(g1,g2)  


# Exploring the Variable Gender
HR_emp_att %>%ggplot(aes(x = Gender, group = Attrition)) +geom_bar(aes(y = ..prop.., fill = factor(..x..)), 
                                                                   stat="count",alpha = 0.7) + geom_text(aes(label = scales::percent(..prop..), y = ..prop.. ),   
                                                                                                         stat= "count",vjust = -.5) +labs(y = "Percentage", fill= "Gender") + facet_grid(~Attrition) + 
  theme_minimal()+theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) + ggtitle("Attrition")            


# Exploring the variable MaritalStatus           
HR_emp_att %>%ggplot(aes(x = MaritalStatus, group = Attrition)) +geom_bar(aes(y = ..prop.., fill = factor(..x..)),
                                                                          stat="count",alpha = 0.7) +geom_text(aes(label = scales::percent(..prop..), y = ..prop.. ),  
                                                                                                               stat= "count",vjust = -.5) +labs(y = "Percentage", fill= "MaritalStatus") + facet_grid(~Attrition) +  
  theme_minimal()+ theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) +ggtitle("Attrition")           

# exploring the Variable MonthlyIncome  
# emp with low monthly salary trend to quit
HR_emp_att %>%ggplot(mapping = aes(x = MonthlyIncome)) + geom_histogram(aes(fill = Attrition), bins=20)+
  labs(x="Monthlt Income", y="Number Attriation")+ ggtitle("Attrition in regards to Monthly Income")    

HR_emp_att %>% filter(Attrition==1) %>%ggplot(mapping = aes(x = MonthlyIncome)) + geom_histogram(aes(fill = Attrition), bins=20)+
  labs(x="Monthlt Income", y="Number Attriation")+ ggtitle("Attrition in regards to Monthly Income")    

# Exploring the Variable OverTime

ggplot(HR_emp_att, aes(y = YearsSinceLastPromotion, x = YearsAtCompany, colour = OverTime)) + geom_jitter(size = 1, alpha = 0.7) + 
  geom_smooth(method = "gam") + facet_wrap(~ Attrition) + ggtitle("Attrition") + scale_colour_manual(values = c("#386cb0","#fdb462")) + 
  theme(plot.title = element_text(hjust = 0.5))

g1 <-HR_emp_att %>%ggplot(aes(x = OverTime, group = Attrition)) + geom_bar(aes(y = ..prop.., fill = factor(..x..)),
                                                                           stat="count",alpha = 0.7) + geom_text(aes(label = scales::percent(..prop..), y = ..prop.. ), 
                                                                                                                 stat= "count",vjust = 0.3) +labs(y = "Percentage", fill= "OverTime") +facet_grid(~Attrition) +    
  theme_minimal()+ theme(legend.position = "none", plot.title = element_text(hjust = 0.3)) +           
  ggtitle("Attrition")         

g2 <-HR_emp_att %>% group_by(OverTime, Attrition) %>% tally() %>% ggplot(aes(x = OverTime, y = n,fill=Attrition)) +         
  geom_bar(stat = "identity") + theme_minimal()+ geom_text(aes(label = n), vjust = -0.3, position = position_dodge(0.9))+
  labs(x="Over time", y="Number Attriation")+ggtitle("Attrition in regards to Over time")

grid.arrange(g1,g2)

# First graph shows Years at company in relation to Years since last promotion, grouped by both attrition and 
# overtime. This is an interesting issue, since a high correlation between these two variables 
# (longer you are in the company, less chance you have to be promoted, so to speak) may mean that people are not
# really growing within the company. However, since this is a simulated dataset we cannot compare it with some 
# norms outside it, so we can compare certain groups within our set, e.g. those who are working overtime and 
# those who are not.
# 
# Here we can note two things. Firstly, there is a relatively higher percentage of people working overtime in the
# group of those who left, an observation confirmed by our barchart. Secondly, while things seem to be going in 
# the right direction for the group of people who still work for the IBM (higher correlation between years since
# last promotion and years at company for those who don't work overtime), the opposite is happening in the other
# group. It seems that there may be a pattern of people leaving because they are not promoted although they work
# hard. This is only an assumption at this point, since the confidence intervals (gray area around the lines) are
# getting wider, meaning there is not that much certainty about this, especially at higher values of X and Y 
# (due to the lack of data).


# Exploring the Variable WorklifeBalance
g1<-HR_emp_att%>%ggplot(aes(x = WorkLifeBalance, group = Attrition)) + geom_bar(aes(y = ..prop.., fill = factor(..x..)), 
                                                                                stat="count", alpha = 0.7) +geom_text(aes(label = scales::percent(..prop..), y = ..prop.. ), 
                                                                                                                      stat= "count", vjust = -.5) +labs(y = "Percentage", fill= "WorkLifeBalance") +
  facet_grid(~Attrition) + theme_minimal()+ theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) + 
  ggtitle("Attrition")

g2<- HR_emp_att %>% group_by(WorkLifeBalance, Attrition) %>% tally() %>% ggplot(aes(x = WorkLifeBalance, y = n,fill=Attrition)) +
  geom_bar(stat = "identity") + theme_minimal()+geom_text(aes(label = n), vjust = -0.5, position = position_dodge(0.9))+
  labs(x="  Work Life Balance", y="Number Attriation") + ggtitle("Attrition in regards to  Work Life Balance")

grid.arrange(g1,g2)

# Exploring the Variables WorkLifeBalance and DistanceFromHome
ggplot(HR_emp_att,aes(x= WorkLifeBalance, y=DistanceFromHome, group = WorkLifeBalance, fill = WorkLifeBalance)) + 
  geom_boxplot(alpha=0.7) + theme(legend.position="none") + facet_wrap(~ Attrition) + ggtitle("Attrition") + 
  theme(plot.title = element_text(hjust = 0.5))
#We can see that there are some patterns in the work-life balance if we compare the two groups within the Attrition variable.

# Those who rated their work-life balance relatively low were commuting from a bit farther away in comparison
# with those who rated their work-life balance as very good. This difference is more pronounced in the group of 
# those who are not in the company anymore, suggesting a possibly influential attrition factor. If the distance 
# is measured in kilometers or miles, this of course doesn't represent a long distance, but since we're dealing
# with a simulated dataset, it might play a role in predicting our attrition class.  
# visualization of Attrition

HR_emp_att %>% group_by(Attrition) %>% tally() %>%ggplot(aes(x = Attrition, y = n,fill=Attrition)) +
  geom_bar(stat = "identity") + theme_minimal()+ labs(x="Attrition", y="Count of Attriation")+
  ggtitle("Attrition")+ geom_text(aes(label = n), vjust = -0.5, position = position_dodge(0.9))

# As it is obvious from the above graph, there is a great imbalance between the two classes and we are encountered an unbalanced
# classification problem. In order to handle this issue we utilize the SMOTE method which can generate a new smoothed data set
# that addresses the class unbalance problem.we will balance the data later now we will build model on row data.
# 




# data preparation

rfData <- HR_emp_att
set.seed(123)
indexes = sample(1:nrow(rfData), size=0.8*nrow(rfData))
RFRaw.train.Data <- rfData[indexes,]
RFRaw.test.Data <- rfData[-indexes,]

# validate dataset
RFRaw.validate.Data <- read.csv('pfm_test')

# Removing non value attributes
RFRaw.validate.Data$EmployeeNumber<- NULL
RFRaw.validate.Data$StandardHours <- NULL
RFRaw.validate.Data$Over18 <- NULL
RFRaw.validate.Data$EmployeeCount <- NULL
cat("Data Set has ",dim(HR_emp_att)[1], " Rows and ", dim(HR_emp_att)[2], " Columns" )

# There are some attributes that are categorical, but in the data set are integer. We have to change them to categorical.
RFRaw.validate.Data$Education <- factor(RFRaw.validate.Data$Education)
RFRaw.validate.Data$EnvironmentSatisfaction <- factor(RFRaw.validate.Data$EnvironmentSatisfaction)
RFRaw.validate.Data$JobInvolvement <- factor(RFRaw.validate.Data$JobInvolvement)
RFRaw.validate.Data$JobLevel <- factor(RFRaw.validate.Data$JobLevel)
RFRaw.validate.Data$JobSatisfaction <- factor(RFRaw.validate.Data$JobSatisfaction)
RFRaw.validate.Data$PerformanceRating <- factor(RFRaw.validate.Data$PerformanceRating)
RFRaw.validate.Data$RelationshipSatisfaction <- factor(RFRaw.validate.Data$RelationshipSatisfaction)
RFRaw.validate.Data$StockOptionLevel <- factor(RFRaw.validate.Data$StockOptionLevel)
RFRaw.validate.Data$WorkLifeBalance <- factor(RFRaw.validate.Data$WorkLifeBalance)

# Building the model

Raw.rf.model <- randomForest(Attrition~.,RFRaw.train.Data, importance=TRUE,ntree=1000)

# plotting variable importance 
varImpPlot(Raw.rf.model)  

# As we see here, Over time, Age, MonthlyIncome, Jobrole and TotalWorkingYears are top five variables.            

Raw.rf.prd <- predict(Raw.rf.model, newdata = RFRaw.test.Data)
Raw.rf.prd <- sapply(Raw.rf.prd, function(x) ifelse(x>0.4,1,0))
confusionMatrix(RFRaw.test.Data$Attrition, Raw.rf.prd)  

Raw.rf.plot<- plot.roc(as.numeric(RFRaw.test.Data$Attrition), as.numeric(Raw.rf.prd),lwd=2, type="b",print.auc=TRUE,col ="blue")   

# Acc = 0.983 which is very good result, We see that the AUC is 0.938  

# Feature Engineering

# Now we want to use some data wrapping to make the results better:
#   
# Making Age Group 18-24 = Young , 25-54=Middle-Age , 54-120= Adult

HR_emp_att$AgeGroup <- as.factor(ifelse(HR_emp_att$Age<=28,"Young", ifelse(HR_emp_att$Age<=38,"Middle-Age","Adult")))
table(HR_emp_att$AgeGroup) 

# as we see the majority of employee are Young

# 2- Creating a new variable Tot.Satisfaction i.e. the total of the satisfaction from Job, Environment,
# JobInvolvement,JobSatisfaction,RelationshipSatisfaction,TotlaSatisfaction etc.
HR_emp_att$Tot.Satisfaction <- 
  as.numeric(HR_emp_att$EnvironmentSatisfaction)+
  as.numeric(HR_emp_att$JobInvolvement)+
  as.numeric(HR_emp_att$JobSatisfaction)+
  as.numeric(HR_emp_att$RelationshipSatisfaction)+
  as.numeric(HR_emp_att$WorkLifeBalance)

summary(HR_emp_att$Tot.Satisfaction)
# 
ggplot(HR_emp_att, aes(x = Tot.Satisfaction, fill = AgeGroup, colour = Attrition)) + geom_density() +
                                                              labs(x="Tot.Satisfaction", y="%")

# 3- Study Years for getting Education Level certificate           
table(HR_emp_att$Education)
# As we see here, there are five Education level
# From high School to PhD (HighSchool=10 years, College=2 years, Bachelor=4 years,Master=2 years,PhD= four years)

HR_emp_att$YearsEducation <-  ifelse(HR_emp_att$Education==1,10,ifelse(HR_emp_att$Education==2,12,
                                                                       ifelse(HR_emp_att$Education==3,16,ifelse(HR_emp_att$Education==4,18,22))))  

table(HR_emp_att$YearsEducation)
# we used culumative years for any level 
# the majority of employee are 16 years education (Bachelor)

HR_emp_att$IncomeLevel <- as.factor(ifelse(HR_emp_att$MonthlyIncome<mean(HR_emp_att$MonthlyIncome),"Low","High"))
table(HR_emp_att$IncomeLevel)

# Let see the Correlation Matrix of Data
corrplot(cor(sapply(HR_emp_att,as.integer)),method = "pie") 

# keep iid, remove high correlate features
HR_emp_att$Education <- NULL
HR_emp_att$Age <- NULL
HR_emp_att$JobLevel<- NULL
HR_emp_att$MonthlyIncome<- NULL

# We can see some of variables are high correlated
# As an Example :
# JobLevel and MonthlyIncome
# Education and YearsEducation
# 
# they cause multicollinearity problem in our data set,so we have to deciede to remove one of them for any group Now we try again our data set with new attributes using Random Forest   

# New Random Forest 
# data preparation
rfData <- HR_emp_att
set.seed(123)
indexes = sample(1:nrow(rfData), size=0.8*nrow(rfData))
RFtrain.Data <- rfData[indexes,]
RFtest.Data <- rfData[-indexes,]

rf.model <- randomForest(Attrition~.,RFtrain.Data, importance=TRUE,ntree=500)
varImpPlot(rf.model)
# Here we see: OverTime, TotalSatisfaction (New Feature), MonthlyIncome, Age and TotalWorkingYears are top five
# variables.

rf.prd <- predict(rf.model, newdata = RFtest.Data)
rf.prd <- sapply(rf.prd,function(x) ifelse(x>0.4,1,0))
confusionMatrix(RFtest.Data$Attrition, rf.prd)
rf.Plot<- plot.roc (as.numeric(RFtest.Data$Attrition), as.numeric(rf.prd),lwd=2, type="b", print.auc=TRUE,col ="blue")

import xgboost as xgb
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# It is better than previous  algorithm with raw data.

# TotalSatisfaction is high important attribute

# Let using other algorithms

## Using Support Vector Machine
svmData <- HR_emp_att
set.seed(123)
indexes = sample(1:nrow(svmData), size=0.8*nrow(svmData))
SVMtrain.Data <- svmData[indexes,]
SVMtest.Data <- svmData[-indexes,]
tuned <- tune(svm,factor(Attrition)~.,data = SVMtrain.Data)
svm.model <- svm(SVMtrain.Data$Attrition~., data=SVMtrain.Data
                 ,type="C-classification", gamma=tuned$best.model$gamma
                 ,cost=tuned$best.model$cost
                 ,kernel="radial")
svm.prd <- predict(svm.model,newdata=SVMtest.Data)
confusionMatrix(svm.prd,SVMtest.Data$Attrition) 

svm.plot <-plot.roc (as.numeric(SVMtest.Data$Attrition), as.numeric(svm.prd),lwd=2, type="b", print.auc=TRUE,col ="blue")                 

# as we see, in compare to RF, Accuracy is lower to 0.9167 and AUC to 0.698 which is not better than RF.

# There is no False Negative! and a lot of False Positive!  


# Decision Tree
# Here Dtree will be investigated and compared to other approaches. DTree selected because it is a very
# good algorithm for interpretion for non-technical.

DtData <- HR_emp_att
set.seed(123)
indexes = sample(1:nrow(DtData), size=0.8*nrow(DtData))
DTtrain.Data <- DtData[indexes,]
DTtest.Data <- DtData[-indexes,]

dtree.model <- tree::tree (Attrition ~., data = DTtrain.Data)
plot(dtree.model)
text(dtree.model, all = T)
dtree.prd <- predict(dtree.model, DTtest.Data, type = "class")
confusionMatrix(dtree.prd,DTtest.Data$Attrition)
dtree.plot <- plot.roc (as.numeric(DTtest.Data$Attrition), as.numeric(dtree.prd),lwd=2, type="b", print.auc=TRUE, col ="blue")

# Not very nice result! Accuracy is 0.8724  where AUC is 0.630 which is not better than SVM 


# Exterme Gradient Boost

set.seed(123)
xgbData <- HR_emp_att
indexes <- sample(1:nrow(xgbData), size=0.8*nrow(xgbData))
XGBtrain.Data <- xgbData[indexes,]
XGBtest.Data <- xgbData[-indexes,]

formula = Attrition~.
fitControl <- trainControl(method="cv", number = 3,classProbs = TRUE )
xgbGrid <- expand.grid(nrounds = 50,
                       max_depth = 12,
                       eta = .03,
                       gamma = 0.01,
                       colsample_bytree = .7,
                       min_child_weight = 1,
                       subsample = 0.9
)  

XGB.model <- train(formula, data = XGBtrain.Data,
                   method = "xgbTree"
                   ,trControl = fitControl
                   , verbose=0
                   , maximize=FALSE
                   ,tuneGrid = xgbGrid
)

importance <- varImp(XGB.model)
varImportance <- data.frame(Variables = row.names(importance[[1]]), 
                            Importance = round(importance[[1]]$Overall,2))
# Create a rank variable based on importance of variables
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance)) +
  geom_bar(stat='identity',colour="white", fill = "lightgreen") +
  geom_text(aes(x = Variables, y = 1, label = Rank),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Variables', title = 'Relative Variable Importance') +
  coord_flip() + 
  theme_bw()
# As we see above: MonthlyIncome, DailyRate, OvertimeYes, TotalSatisfaction and Age are top five attributes.
XGB.prd <- predict(XGB.model,XGBtest.Data)
XGB.prd <- sapply(XGB.prd , function(x) ifelse(x>.4,1,0))
confusionMatrix(XGB.prd, XGBtest.Data$Attrition)
XGB.plot <- plot.roc (as.numeric(XGBtest.Data$Attrition), as.numeric(XGB.prd),lwd=2, type="b", print.auc=TRUE,col ="blue")
# As we see the ACC is 0.9524 which is very good result.
# Perfect: very good result is in Balanced Accuracy which is 0.8375 better than Svm and Decision tree




# Solving Unbalanced Problem
# As we mentioned before there is unbalanced problem in class label. There are some approaches to solve the
# problem. here we use SMOT method.

Classcount = table(HR_emp_att$Attrition)
Classcount
# Over Sampling
over = ( (0.6 * max(Classcount)) - min(Classcount) ) / min(Classcount)
# Under Sampling
under = (0.4 * max(Classcount)) / (min(Classcount) * over)
over = round(over, 1) * 100
under = round(under, 1) * 100
#Generate the balanced data set

BalancedData = SMOTE(Attrition~., HR_emp_att, perc.over = 
                       over, k = 5, perc.under = under)
# let check the output of the Balancing
BalancedData %>%
  group_by(Attrition) %>%
  tally() %>%
  ggplot(aes(x = Attrition, y = n,fill=Attrition)) +
  geom_bar(stat = "identity") +
  theme_minimal()+
  labs(x="Attrition", y="Count of Attriation")+
  ggtitle("Attrition")+
  geom_text(aes(label = n), vjust = -0.5, position = position_dodge(0.9))

# Now we try to run again Random Forest with the Balanced Data
set.seed(123)
RF_Bal_Data <- BalancedData
indexes = sample(1:nrow(RF_Bal_Data), size=0.8*nrow(RF_Bal_Data))
RF_BLtrain.Data <- RF_Bal_Data[indexes,]
RF_BLtest.Data <- RF_Bal_Data[-indexes,]

RF_BAL.model <- randomForest(Attrition~.,RF_BLtrain.Data, importance=TRUE,ntree=500)
varImpPlot(RF_BAL.model)
RF_BAL.prd <- predict(RF_BAL.model, newdata = RF_BLtest.Data)
confusionMatrix(RF_BLtest.Data$Attrition, RF_BAL.prd)
RF_BAL.Plot<- plot.roc (as.numeric(RF_BLtest.Data$Attrition), as.numeric(RF_BAL.prd),lwd=2, type="b", print.auc=TRUE,col ="blue")




# Now we try to run again XGBoost with the Balanced Data
set.seed(123)
xgbData <- BalancedData
indexes = sample(1:nrow(xgbData), size=0.8*nrow(xgbData))
BLtrain.Data <- xgbData[indexes,]
BLtest.Data <- xgbData[-indexes,]

formula = Attrition~.
fitControl <- trainControl(method="cv", number = 3,classProbs = TRUE )
xgbGrid <- expand.grid(nrounds = 500,
                       max_depth = 20,
                       eta = .03,
                       gamma = 0.01,
                       colsample_bytree = .7,
                       min_child_weight = 1,
                       subsample = 0.9
)



importance <- varImp(XGB.model)
varImportance <- data.frame(Variables = row.names(importance[[1]]), 
                            Importance = round(importance[[1]]$Overall,2))
# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance)) +
  geom_bar(stat='identity',colour="white", fill = "lightgreen") +
  geom_text(aes(x = Variables, y = 1, label = Rank),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  labs(x = 'Variables', title = 'Relative Variable Importance') +
  coord_flip() + 
  theme_bw()

NewXGB.prd <- predict(XGB.model,BLtest.Data)
NewXGB.prd <- sapply(NewXGB.prd , function(x) ifelse(x>.5,1,0))
confusionMatrix(NewXGB.prd, BLtest.Data$Attrition)

XGB.plot <- plot.roc (as.numeric(BLtest.Data$Attrition), as.numeric(NewXGB.prd),lwd=2, type="b", print.auc=TRUE, col ="blue")
XGB.plot

# Congratulation !
#   Excelent Results:
#   Accuracy : more than 90% !!!!!
#   AUC about 0.89

# Here we plot all the approaches in one plot:

par(mfrow=c(3,3))
plot.roc (as.numeric(RFRaw.test.Data$Attrition), as.numeric(Raw.rf.prd), main="Raw Data Random Forest",lwd=2, type="b", print.auc=TRUE, col ="seagreen4")
plot.roc (as.numeric(RFtest.Data$Attrition), as.numeric(rf.prd), main=" NEW Data Random Forest",lwd=2, type="b", print.auc=TRUE, col ="seagreen")
RF_BAL.Plot<- plot.roc (as.numeric(RF_BLtest.Data$Attrition), as.numeric(RF_BAL.prd),main="Balanced Data Random Forest",lwd=2, type="b", print.auc=TRUE,col ="blue")
plot.roc (as.numeric(DTtest.Data$Attrition), as.numeric(dtree.prd), main="DTree",lwd=2, type="b", print.auc=TRUE, col ="brown")
plot.roc (as.numeric(SVMtest.Data$Attrition), as.numeric(svm.prd),main="SVM",lwd=2, type="b", print.auc=TRUE, col ="red")
plot.roc (as.numeric(XGBtest.Data$Attrition), as.numeric(XGB.prd),main="XGBoost",lwd=2, type="b", print.auc=TRUE, col ="blue")
plot.roc (as.numeric(BLtest.Data$Attrition), as.numeric(NewXGB.prd),main=" Balanced Data XGBoost",lwd=2, type="b", print.auc=TRUE, col ="green")








