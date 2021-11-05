# Michael Prappas
# EMIS 8331 - Adv. Data Mining
# Final Project

library(rpart)
library(rpart.plot)
library(caret)
library(FSelector)

setwd("/Users/Michael/Documents/SMU - Senior/EMIS 8331 - Advanced Data Mining/Final Project")

#
# LOAD DATA FROM PYTHON CODE
#

news_sentimental <- read.csv("news_sentimental.csv")
article_terms_500 <- read.csv("article_terms_500.csv", header = FALSE)
vocabulary_sel_500 <- read.csv("vocabulary_sel_500.csv", header = FALSE)

# add X identifier to article_terms_500 (sparse matrix) to not get confused
article_terms_500 <- cbind(news_sentimental$X, article_terms_500)


#
# FIND TOP SENTIMENTAL FEATURES
#

top_sent_features <- chi.squared(label ~ ., data = news_sentimental[,c(-1, -2, -3)])
top_sent_features
# the ones with 0: all the title ones, all the first ones except first_neu, last_comp
# most important feature: most_pos
# second-most important: most_subj
# third: avg_comp

length(top_sent_features$attr_importance)
# 30 features
length(which(top_sent_features$attr_importance == 0.0))
# 12 unimportant
# use 18 important features

#
# SENTIMENTAL-ONLY CLASSIFIERS
# GOAL: FIND IDEAL NUMBER OF FEATURES
# doing 10-fold cross-validation
# 10-fold cross-validation code from Hahsler, EMIS 7331
#

index <- 1:nrow(news_sentimental)
index <- sample(index)
fold <- rep(1:10, each=nrow(news_sentimental)/10)[1:nrow(news_sentimental)]
folds <- split(index, fold) 

accuracy <- function(truth, prediction) {
  tbl <- table(truth, prediction)
  sum(diag(tbl))/sum(tbl)
}

# start with all features
accs <- vector(mode = "numeric")
for (i in 1:length(folds)) {
  all_feature_tree <- rpart(label ~ ., 
                            data = news_sentimental[-folds[[i]],c(-1, -2, -3)])
  accs[i] <- accuracy(news_sentimental[folds[[i]],]$label, predict(all_feature_tree, news_sentimental[folds[[i]],c(-1, -2, -3)], type="class"))
}
mean(accs)
# mean accuracy = 0.6406003

# top 4 features
for (i in 1:length(folds)) {
  four_feature_tree <- rpart(label ~ most_pos + most_subj + avg_comp + low_pol, 
                             data = news_sentimental[-folds[[i]],c(-1, -2, -3)])
  accs[i] <- accuracy(news_sentimental[folds[[i]],]$label, predict(four_feature_tree, news_sentimental[folds[[i]],c(-1, -2, -3)], type="class"))
}
mean(accs)
# mean accuracy = 0.6

# top 6 features
for (i in 1:length(folds)) {
  six_feature_tree <- rpart(label ~ most_pos + most_subj + avg_comp + low_pol + last_neu + avg_neg, 
                            data = news_sentimental[-folds[[i]],c(-1, -2, -3)])
  accs[i] <- accuracy(news_sentimental[folds[[i]],]$label, predict(six_feature_tree, news_sentimental[folds[[i]],c(-1, -2, -3)], type="class"))
}
mean(accs)
# mean accuracy = 0.6120063

# top 8 features
for (i in 1:length(folds)) {
  eight_feature_tree <- rpart(label ~ most_pos + most_subj + avg_comp + low_pol + last_neu + avg_neg + avg_neu + high_pol, 
                              data = news_sentimental[-folds[[i]],c(-1, -2, -3)])
  accs[i] <- accuracy(news_sentimental[folds[[i]],]$label, predict(eight_feature_tree, news_sentimental[folds[[i]],c(-1, -2, -3)], type="class"))
}
mean(accs)
# mean accuracy = 0.6328594

# top 10 features
for (i in 1:length(folds)) {
  ten_feature_tree <- rpart(label ~ most_pos + most_subj + avg_comp + low_pol + last_neu + avg_neg + avg_neu + high_pol + most_neg + last_pos, 
                            data = news_sentimental[-folds[[i]],c(-1, -2, -3)])
  accs[i] <- accuracy(news_sentimental[folds[[i]],]$label, predict(ten_feature_tree, news_sentimental[folds[[i]],c(-1, -2, -3)], type="class"))
}
mean(accs)
# mean accuracy = 0.6372828

# top 12 features
for (i in 1:length(folds)) {
  twelve_feature_tree <- rpart(label ~ most_pos + most_subj + avg_comp + low_pol + last_neu + avg_neg + avg_neu + high_pol + most_neg + last_pos + avg_pol + most_neu, 
                               data = news_sentimental[-folds[[i]],c(-1, -2, -3)])
  accs[i] <- accuracy(news_sentimental[folds[[i]],]$label, predict(twelve_feature_tree, news_sentimental[folds[[i]],c(-1, -2, -3)], type="class"))
}
mean(accs)
# mean accuracy = 0.6369668

# top 14 features
for (i in 1:length(folds)) {
  fourtn_feature_tree <- rpart(label ~ most_pos + most_subj + avg_comp + low_pol + last_neu + avg_neg + avg_neu + high_pol + most_neg + last_pos + avg_pol + most_neu + avg_pos + last_subj, 
                               data = news_sentimental[-folds[[i]],c(-1, -2, -3)])
  accs[i] <- accuracy(news_sentimental[folds[[i]],]$label, predict(fourtn_feature_tree, news_sentimental[folds[[i]],c(-1, -2, -3)], type="class"))
}
mean(accs)
# mean accuracy = 0.6406003

# top 16 features
for (i in 1:length(folds)) {
  sixtn_feature_tree <- rpart(label ~ most_pos + most_subj + avg_comp + low_pol + last_neu + avg_neg + avg_neu + high_pol + most_neg + last_pos + avg_pol + most_neu + avg_pos + last_subj + last_pol + avg_subj, 
                              data = news_sentimental[-folds[[i]],c(-1, -2, -3)])
  accs[i] <- accuracy(news_sentimental[folds[[i]],]$label, predict(sixtn_feature_tree, news_sentimental[folds[[i]],c(-1, -2, -3)], type="class"))
}
mean(accs)
# mean accuracy = 0.6406003

# using 14 features: no better accuracy than by going up to 18

#
# FINDING BEST CLASSIFIERS TO USE WITH SENTIMENTAL DATA
# again using code from Hahsler, EMIS 7331
#

train <- createFolds(news_sentimental$label, k=10)

# replace NA with 0 to handle empty articles
# some classifiers don't like NAs in the data
news_sentimental[is.na(news_sentimental)] <- 0

# CART
cart_tree <- train(label ~ most_pos + most_subj + avg_comp + low_pol + last_neu + avg_neg + avg_neu + high_pol + most_neg + last_pos + avg_pol + most_neu + avg_pos + last_subj, 
                   method = "rpart", data = news_sentimental,
                   trControl = trainControl(
                     method = "cv", indexOut = train))
cart_tree
# max accuracy: 0.6397823
# max kappa: 0.27948213

# conditional inference tree
cond_tree <- train(label ~ most_pos + most_subj + avg_comp + low_pol + last_neu + avg_neg + avg_neu + high_pol + most_neg + last_pos + avg_pol + most_neu + avg_pos + last_subj, 
                   method = "ctree", data = news_sentimental,
                   trControl = trainControl(
                     method = "cv", indexOut = train))
cond_tree
# max accuracy: 0.7452250
# max kappa: 0.4904563

# C4.5
c4.5_tree <- train(label ~ most_pos + most_subj + avg_comp + low_pol + last_neu + avg_neg + avg_neu + high_pol + most_neg + last_pos + avg_pol + most_neu + avg_pos + last_subj, 
                   method = "J48", data = news_sentimental,
                   trControl = trainControl(
                     method = "cv", indexOut = train))
c4.5_tree
# max accuracy: 0.7136430
# max kappa: 0.4272483
c4.5_tree$finalModel

# PART (rule-based)
part_tree <- train(label ~ most_pos + most_subj + avg_comp + low_pol + last_neu + avg_neg + avg_neu + high_pol + most_neg + last_pos + avg_pol + most_neu + avg_pos + last_subj, 
                   method = "PART", data = news_sentimental,
                   trControl = trainControl(
                     method = "cv", indexOut = train))
part_tree
# max accuracy: 0.6786100
# max kappa: 0.3571448

# Linear Support Vector Machines
lsvm_clfr <- train(label ~ most_pos + most_subj + avg_comp + low_pol + last_neu + avg_neg + avg_neu + high_pol + most_neg + last_pos + avg_pol + most_neu + avg_pos + last_subj, 
                   method = "svmLinear", data = news_sentimental,
                   trControl = trainControl(
                     method = "cv", indexOut = train))
lsvm_clfr
# max accuracy: 0.6069461
# max kappa: 0.2137527

# Artificial Neural Network
nnet_clfr <- train(label ~ most_pos + most_subj + avg_comp + low_pol + last_neu + avg_neg + avg_neu + high_pol + most_neg + last_pos + avg_pol + most_neu + avg_pos + last_subj, 
                   method = "nnet", data = news_sentimental,
                   trControl = trainControl(
                     method = "cv", indexOut = train))
nnet_clfr
# max accuracy: 0.6735694
# max kappa: 0.3470616

# Random Forest
rfst_clfr <- train(label ~ most_pos + most_subj + avg_comp + low_pol + last_neu + avg_neg + avg_neu + high_pol + most_neg + last_pos + avg_pol + most_neu + avg_pos + last_subj, 
                   method = "rf", data = news_sentimental,
                   trControl = trainControl(
                     method = "cv", indexOut = train))
rfst_clfr
# max accuracy: 0.9695329
# max kappa: 0.9390653

# Random forest is far and away the best
# best of the other ones: ctree
# runner-up: C4.5

save(rfst_clfr, file = "rfst_clfr.RData")

pred <- predict(rfst_clfr)
pred_plus_actual <- as.data.frame(cbind(pred, news_sentimental$label))
wrong_ID <- which(pred_plus_actual$pred != pred_plus_actual$V2)
# 347  752  757 1269 1933 2115 2146 2262 2651 3051 3532 3538 3578 4033 4048 4116 4265 4397 4661 5094 5315
# 5331 5764 5932 6134 6172

sort(news_sentimental$X[wrong_ID], decreasing = FALSE)

which(news_sentimental$label[wrong_ID] == "REAL")
# 8