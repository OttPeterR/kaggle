# r
# from datacamp tutorial

# remember to run this in r console:
# install.packages('randomForest')
library('randomForest')
set.seed(0)

test <- read.csv("test.csv")
train <- read.csv("train.csv")


# Survival rates in absolute numbers
#   0   1
# 549 342
# table(train$Survived)

# Survival rates in proportions
#         0         1
# 0.6161616 0.3838384
# prop.table(table(train$Survived))
  
# Two-way comparison: Sex and Survived
#            0   1
#   female  81 233
#   male   468 109
# table(train$Sex, train$Survived)

# Two-way comparison: row-wise proportions
#              0         1
# female 0.2579618 0.7420382
# male   0.8110919 0.1889081
# prop.table(table(train$Sex, train$Survived), 1)

# making child column
train$Child <- NA
train$Child[train$Age < 18] <- 1
train$Child[train$Age >= 18] <- 0

test$Child <- NA
test$Child[train$Age < 18] <- 1
test$Child[train$Age >= 18] <- 0
test$Survived <- NA

str(test)

#making a cumulative dataset
all_data = rbind(train, test)

all_data$Embarked[c(62, 830)] <- "S"
all_data$Embarked <- factor(all_data$Embarked)


all_data$Fare[1044] <- median(all_data$Fare, na.rm = TRUE)


predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + family_size,
                       data = all_data[!is.na(all_data$Age),], method = "anova")
all_data$Age[is.na(all_data$Age)] <- predict(predicted_age, all_data[is.na(all_data$Age),])

train <- all_data[1:891,]
test <- all_data[892:1309,]

my_forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, ntree=1000, data=train, importance=TRUE)
my_prediction <- predict(my_forest, test)
my_solution <- data.frame(PassengerId = test$PassengerId, Survived=my_prediction)
write.csv(my_solution, file="submission.csv", row.names=FALSE)