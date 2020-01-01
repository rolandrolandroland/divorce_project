library(readr)
library(caret)
library(dplyr)
library(randomForest)
library(Rborist)
library(gam)
library(rmarkdown)

# explore data
read_lines("divorce/divorce.csv", n_max = 20)
# create object data with the data
data = read_csv2("divorce/divorce.csv")
head(data)
nrow(data)
View(data)
sum(data$Class==1) # Divorced
sum(data$Class==0) # Married
# It appears that 4 is the max value, which corresponds to 'no' while 0 corresponds to 'yes'
sapply(data, max)

# Create Training set with 80% of data
set.seed(1, sample.kind = "Rounding")
train_index = createDataPartition(data$Class, times = 1, p = 0.8, list = FALSE)
train_set = data[train_index,]
not_train_set = data[-train_index,]

# The data needs to be of class factor
level_train = droplevels(train_set)
level_train$Class = as.factor(level_train$Class)
class(level_train$Class)

# Create ensemble model
models <- c("glm", "lda", "naive_bayes", "svmLinear", "knn", "gamLoess", "rf", "adaboost")
fits = lapply(models, function(model){
  print(model)
  train(Class~., method = model, data = level_train)
})
# name each fit
names(fits) = models

# Create Prediction using test set
mat_pred = sapply(fits, function(fit){
  predict(fit, newdata =not_train_set)
})

# Create Mode Function- prediction will be mode of ensemble results
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
# The prediction will be the most common result from all of the models AKA the mode
prediction = as.numeric(apply(mat_pred, 1, getmode))
# Determine if all predictions are correct
identical(prediction, not_train_set$Class)
# Already have 100% accuracy 


# try running monte carlo

couples = 1:170
# get rid of adaboost, it takes too long for a monte carlo simulation
models_2 <- c("glm", "lda", "naive_bayes", "svmLinear", "knn", "gamLoess", "rf")
#### WARNING the below function takes a long time to run!!
monte = replicate(1000, {
  trainer = sample(couples, 136, replace = FALSE) # this will run on 80% of the data
  tester = data[-trainer,]
  trainer = data[trainer, ]
  trainer = droplevels(trainer)
  trainer$Class = as.factor(trainer$Class)
  fitter = lapply(models_2, function(model){
    train(Class~., method = model, data = trainer)
  })
  pred = sapply(fitter, function(fit){
    predict(fit, newdata = tester)
  })
  vec =as.numeric(apply(pred, 1, getmode))
  mean(vec == tester$Class)
})
monte
mean(monte)
sd(monte)
sum(monte ==1)

# Now I want to find the most important predictors
# For each predictor, predict divorced if answer is greater than each number 0 through 4
over = sapply(nums, function(number){
  colMeans(as.data.frame(data >= number) == data$Class)
})

# set column names corresponding to the prediction
colnames(over) = c("0", "1", "2", "3", "4")

# get rid of column 55, the column that contains whether or not the couple is divorced since it is not a predictor
over = over[-55,] 
# Set row names
rownames(over) = paste("Atr", 1:54, sep = "")
over_tbl =as_tibble(over)
key_column =  paste("Atr", 1:54, sep = "")
row_maxs = apply(over, 1, max)
max_index = apply(over, 1, which.max)

sort(row_maxs, decreasing = T)

over_2 = cbind(over, row_maxs, max_index)
over_2_tbl = as_tibble(over_2) 
over_2_tbl = cbind(key_column, over_2_tbl) %>% as_tibble()
colnames(over_2_tbl) = c("Predictor", '0', '1', '2', '3', '4', "Max", "Max Index")
over_2_tbl
over_2_tbl %>% top_n(5, Max) %>% arrange(desc(Max))
# So now I know that the four most important indicators according to the study are 18, 11, 17, 19
# Now, try and rerun the algorithm, but only use the four indicators above
# fit model
fits_2 = lapply(models, function(model){
  print(model)
  train(Class~Atr18+Atr11+Atr17+Atr19, method = model, data = level_train)
})
# make predictions 
mat_pred_2 = sapply(fits_2, function(fit){
  predict(fit, newdata =not_train_set)
})
mat_pred_2
prediction_2 = as.numeric(apply(mat_pred_2, 1, getmode))
identical(prediction_2, not_train_set$Class)
# Perfect accuracy

# simulate only using 4 random predictors as well as the 4 top predictors and compare them
monte_2 = replicate(1000, {
  trainer = sample(couples, 136, replace = FALSE) # this will run on 80% of the data
  predictors = sample(1:54, 4, replace= FALSE) # use 4 random predictors
  tester = data[-trainer,]
  trainer = data[trainer, ]
  trainer = droplevels(trainer)
  trainer$Class = as.factor(trainer$Class)
  #class(trainer$Atr18)
  #class(trainer[,predictors[1]])
  #print(predictors)
  #print(trainer[,predictors[1]])
  trainer$Atr1 = as.numeric(unlist(as.list(trainer[,predictors[1]]))) # make them suitable for the train function
  trainer$Atr2 = as.numeric(unlist(as.list(trainer[,predictors[2]]))) # I am replacing the old Atr2 with the value of a random Atr
  trainer$Atr3 = as.numeric(unlist(as.list(trainer[,predictors[3]]))) # this is my solution to the train() function being picky about
  trainer$Atr4 = as.numeric(unlist(as.list(trainer[,predictors[4]]))) # the format of its input. 
  #print(trainer$Atr1)
  fitter = lapply(models_2, function(model){
    train(Class~Atr1 + Atr2 + Atr3 + Atr4, 
          method = model, data = trainer)
  })
  fitter_2 = lapply(models_2, function(model){
    train(Class~Atr11 + Atr17 + Atr18 +Atr19, method = model, data = trainer)
  })
  pred = sapply(fitter, function(fit){
    predict(fit, newdata = tester)
  })
  pred_2 = sapply(fitter_2, function(fit){
    predict(fit, newdata = tester)
  })
  vec =as.numeric(apply(pred, 1, getmode))
  m_1 =(mean(vec == tester$Class))
  vec_2 = as.numeric(apply(pred_2, 1, getmode))
  m_2 =(mean(vec_2==tester$Class))
  print(m_1)
  print(m_2)
  return(c(m_1, m_2))
})
monte_2
mean(monte_2[2,])
mean(monte_2[1,])
sd(monte_2[1,])
sd(monte_2[2,])

sum(monte_2[1,]==1)


