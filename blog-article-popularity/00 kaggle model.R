#######################################
# Kaggle Competition - Analytics Edge
#
# Brian Scherer
# Spring, 2015
#######################################

library(tm)             # Text analytics
library(caTools)        # Splitting of data into train & test sets
library(randomForest)   # randomForest modeling
library(ROCR)           # model evaluation

# Mode Settings
# 0 ==> Internal assessment mode: Model developed/tested using training data only, AUC reported.
# 1 ==> External assessment mode: Kaggle submission prepared (w/ both training & test data).
Mode = 1

# Get data if it doesn't already exist.
if (!exists("NYTimesBlogTrain"))
  {
    NYTimesBlogTrain <- read.csv("NYTimesBlogTrain.csv", stringsAsFactors=FALSE)
    NYTimesBlogTest  <- read.csv("NYTimesBlogTest.csv",  stringsAsFactors=FALSE)
  }

if (Mode == 1) { 
  # Set "Train" and "Test"
  Train <- NYTimesBlogTrain
  Test  <- NYTimesBlogTest
  } else {
  set.seed(7)
  spl = sample.split(NYTimesBlogTrain$Popular, SplitRatio = 0.65)
  Train = subset(NYTimesBlogTrain, spl == TRUE)
  Test = subset(NYTimesBlogTrain, spl == FALSE)
}


###########################################################
#
# Pre-processing
#
###########################################################


#Fill in missing values
for (i in 1:nrow(Train))
{ 
  if(nchar(Train$NewsDesk[i])==0) {
    if(Train$SectionName[i]=="Crosswords/Games") Train$NewsDesk[i]="Business"
    if(Train$SectionName[i]=="Business Day") Train$NewsDesk[i]="Business"
    if(Train$SectionName[i]=="Health") Train$NewsDesk[i]="Science"
    if(Train$SectionName[i]=="Multimedia") Train$NewsDesk[i]="MISSING"
    if(Train$SectionName[i]=="Open") Train$NewsDesk[i]="MISSING"
    if(Train$SectionName[i]=="Opinion") Train$NewsDesk[i]="OpEd"
    if(Train$SectionName[i]=="U.S.") Train$NewsDesk[i]="Styles"
    if(Train$SectionName[i]=="Travel") Train$NewsDesk[i]="Travel"
  }
  
  if(nchar(Train$SectionName[i])==0) Train$SectionName[i]="MISSING"
  if(nchar(Train$SubsectionName[i])==0) Train$SubsectionName[i]="MISSING"
}
for (i in 1:nrow(Test))
{
  if(nchar(Test$NewsDesk[i])==0) {
    if(Test$SectionName[i]=="Crosswords/Games") Test$NewsDesk[i]="Business"
    if(Test$SectionName[i]=="Business Day") Test$NewsDesk[i]="Business"
    if(Test$SectionName[i]=="Health") Test$NewsDesk[i]="Science"
    if(Test$SectionName[i]=="Multimedia") Test$NewsDesk[i]="MISSING"
    if(Test$SectionName[i]=="Open") Test$NewsDesk[i]="MISSING"
    if(Test$SectionName[i]=="Opinion") Test$NewsDesk[i]="OpEd"
    if(Test$SectionName[i]=="U.S.") Test$NewsDesk[i]="Styles"
    if(Test$SectionName[i]=="Travel") Test$NewsDesk[i]="Travel"
  }
  
  if(nchar(Test$SectionName[i])==0) Test$SectionName[i]="MISSING"
  if(nchar(Test$SubsectionName[i])==0) Test$SubsectionName[i]="MISSING"
}


#Convert NewsDesk to factor
temp = c(Train$NewsDesk, Test$NewsDesk)
temp = as.factor(temp)
Train$NewsDesk = head(temp, nrow(Train))
Test$NewsDesk = tail(temp, nrow(Test))
#Convert SectionName to factor
temp = c(Train$SectionName, Test$SectionName)
temp = as.factor(temp)
Train$SectionName = head(temp, nrow(Train))
Test$SectionName = tail(temp, nrow(Test))
#Convert SubsectionName to factor
temp = c(Train$SubsectionName, Test$SubsectionName)
temp = as.factor(temp)
Train$SubsectionName = head(temp, nrow(Train))
Test$SubsectionName = tail(temp, nrow(Test))


#Transform WordCount
Train$WordCount = log(1+Train$WordCount)
Test$WordCount = log(1+Test$WordCount)


# Add weekday & hour to Train and Test
Train$PubDate = strptime(Train$PubDate, "%Y-%m-%d %H:%M:%S")
Train$Weekday = as.factor(Train$PubDate$wday)
Train$Hour = as.factor(Train$PubDate$hour)
Test$PubDate = strptime(Test$PubDate, "%Y-%m-%d %H:%M:%S")
Test$Weekday = as.factor(Test$PubDate$wday)
Test$Hour = as.factor(Test$PubDate$hour)


# Headline bag of words
corpus = Corpus(VectorSource(c(Train$Headline, Test$Headline)))
corpus = tm_map(corpus, tolower)
corpus = tm_map(corpus, PlainTextDocument)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords("english"))
corpus = tm_map(corpus, stemDocument)
dtm = DocumentTermMatrix(corpus)
sparse = removeSparseTerms(dtm, 0.99)
Words = as.data.frame(as.matrix(sparse))
colnames(Words) = make.names(colnames(Words))
colnames(Words) = paste("H_", colnames(Words), sep="")
Headline.Train = head(Words, nrow(Train))
Headline.Test = tail(Words, nrow(Test))

# Abstract bag of words
corpus = Corpus(VectorSource(c(Train$Abstract, Test$Abstract)))
corpus = tm_map(corpus, tolower)
corpus = tm_map(corpus, PlainTextDocument)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords("english"))
corpus = tm_map(corpus, stemDocument)
dtm = DocumentTermMatrix(corpus)
sparse = removeSparseTerms(dtm, 0.99)
Words = as.data.frame(as.matrix(sparse))
colnames(Words) = make.names(colnames(Words))
colnames(Words) = paste("A_", colnames(Words), sep="")
Abstract.Train = head(Words, nrow(Train))
Abstract.Test = tail(Words, nrow(Test))


#Pull words from Headline
grepl.Headline.Train = grepl("does |is |has |how |what |why |who |where |when |which |\\?", Train$Headline, ignore.case=T)
grepl.Headline.Test = grepl("does |is |has |how |what |why |who |where |when |which |\\?", Test$Headline, ignore.case=T)

#Pull words from Abstract
grepl.Abstract.Train = grepl("does |is |has |how |what |why |who |where |when |which |\\?", Train$Abstract, ignore.case=T)
grepl.Abstract.Test = grepl("does |is |has |how |what |why |who |where |when |which |\\?", Test$Abstract, ignore.case=T)

###########################################################
#
# Create final training and test sets
#
###########################################################

# Create Final Training and Testing sets
Final.Train = cbind(Headline.Train, Abstract.Train)
Final.Test  = cbind(Headline.Test, Abstract.Test)
#Final.Train = Headline.Train
#Final.Test = Headline.Test


# Add variables back to training set
Final.Train$NewsDesk = Train$NewsDesk
Final.Train$SectionName = Train$SectionName
Final.Train$SubsectionName = Train$SubsectionName
Final.Train$WordCount = Train$WordCount
Final.Train$Weekday = Train$Weekday
Final.Train$Hour = Final.Train$Hour
Final.Train$grepl.Headline = as.factor(grepl.Headline.Train)
Final.Train$grepl.Abstract = as.factor(grepl.Abstract.Train)
Final.Train$Popular = Train$Popular

# Add variables back to testing set
Final.Test$NewsDesk = Test$NewsDesk
Final.Test$SectionName = Test$SectionName
Final.Test$SubsectionName = Test$SubsectionName
Final.Test$WordCount = Test$WordCount
Final.Test$Weekday = Test$Weekday
Final.Test$Hour = Test$Hour
Final.Test$grepl.Headline = as.factor(grepl.Headline.Test)
Final.Test$grepl.Abstract = as.factor(grepl.Abstract.Test)

###########################################################
#
# Mode == 0:  Internal Assessment (not submitted to Kaggle)
#
###########################################################

if (Mode == 0) {
  
set.seed(7)
rF = randomForest(Popular ~ ., data=Final.Train, importance=TRUE)

Final.Test$Popular = Test$Popular

set.seed(7)
predictROC = predict(rF, newdata=Final.Test, type="response")
pred = prediction(predictROC, Final.Test$Popular)
perf = performance(pred, "tpr", "fpr")
plot(perf)
print(as.numeric(performance(pred, "auc")@y.values))

}


###########################################################
#
# Mode == 1:  Final Model Build & Kaggle Submission
#
###########################################################

if (Mode == 1) {
  
  # Create rF model
  set.seed(7)
  rF = randomForest(Popular ~ ., data=Final.Train, importance=TRUE)
  
  # Predictions on test set:
  PredTest = predict(rF, newdata=Final.Test, type="response")
   
  # Prepare our submission file for Kaggle:
  MySubmission = data.frame(UniqueID = Test$UniqueID, Probability1 = PredTest)
  write.csv(MySubmission, "00 kaggle model (05).csv", row.names=FALSE)
  
}