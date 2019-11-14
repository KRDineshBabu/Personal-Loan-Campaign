setwd ("C:\\Users\\DineshBabu\\Dropbox\\Great Learning\\Machine learning\\Week 4")

loan_dataset <- read.csv("Personal Loan Campaign-dataset.csv", header = TRUE)


#install.packages(c("questionr")
#install.packages("FactoMineR")
#install.packages("factoextra")
#install.packages("GPArotation")
#install.packages("psych ")
#install.packages("DataExplorer")
library(questionr)
library(FactoMineR)
library(factoextra)
library(GPArotation)
library(psych)
library(DataExplorer)


describe(loan_dataset[,-c(1,11)])
summary(loan_dataset)
str(loan_dataset)
nrow(loan_dataset)
ncol(loan_dataset)
head(loan_dataset)
dimnames(loan_dataset)

plot_missing(loan_dataset[,-c(1,11)])
plot_histogram(loan_dataset[,-c(1,11)])

png( 'correlation.png', width = 800, height =  800 )
plot_correlation(loan_dataset[,-c(1,11)])
dev.off()

corr <- cor(loan_dataset[,-c(1,2, 4, 6,7,10,11)])
KMO(corr)

imp<-lapply(loan_dataset[,-c(1,2, 4, 6,7,10,11)],function(x) t.test(x,data = loan_dataset,alternative="two.sided")$p.value)
imp<-as.data.frame(imp)
imp_sort<-imp[order(imp,decreasing = TRUE)]
imp_sort_key<-imp_sort[,imp_sort<0.05]
imp_sort_key

attach(loan_dataset)

loan_dataset_factor <-  subset(loan_dataset, select = c(2,4, 6,7,10, SCR,HOLDING_PERIOD, LEN_OF_RLTN_IN_MNTH,
                                                        AMT_OTH_BK_ATM_USG_CHGS, AMT_MIN_BAL_NMC_CHGS, NO_OF_MOB_DR_TXNS,
                                                        NO_OF_IW_CHQ_BNC_TXNS, NO_OF_OW_CHQ_BNC_TXNS, FLG_HAS_NOMINEE, FLG_HAS_OLD_LOAN))

plot_correlation(loan_dataset_factor)


str(loan_dataset_factor)
summary(loan_dataset_factor)

library(randomForest)
library(caTools)
library(rpart)
library(rattle)
library(ROCR)
library(ineq)
library(rpart.plot)
library(RColorBrewer)
library(data.table)
library(caret)
library(ROSE)
loan_dataset_factor$TARGET <- as.factor(loan_dataset_factor$TARGET)
seed_no=123
set.seed(seed_no)
split <- sample.split(loan_dataset_factor, SplitRatio = 0.7)
training_data <- subset(loan_dataset_factor, split == TRUE)
test_data <- subset(loan_dataset_factor, split==FALSE)
str(training_data)
table(training_data$TARGET)

(sum(training_data$TARGET == 1)/ (sum(training_data$TARGET ==0)+ sum(training_data$TARGET == 1))) *100
(sum(test_data$TARGET == 1)/ (sum(test_data$TARGET ==0)+ sum(test_data$TARGET == 1))) *100


#CART model and respected codes 
set.seed(seed_no)

rpart.ctrl <- rpart.control(minsplit = 25, minbucket = 20, cp =0.00, xval = 50)

cart_model <- rpart(formula =TARGET ~ ., data = training_data[,1:15], 
                    method = "class", control = rpart.ctrl, model = TRUE)
cart_model

rpart.plot(cart_model)

cp_cart <- cart_model$cptable[which.min(cart_model$cptable[,"xerror"]),"CP"]

cp_cart

printcp(cart_model)
plotcp(cart_model)

varImp(cart_model)

cart_model <- prune(cart_model, cp = cp_cart)

fancyRpartPlot(cart_model)



training_data$predict_class_cart <- predict(cart_model, training_data[,1:15] , type = "class")

training_data$predict_score_cart <- predict(cart_model, training_data[,1:15], type = "prob" )


# confusion matrix using "caret" package 

confusion_matrix_train <-confusionMatrix(training_data$TARGET, training_data$predict_class_cart)
confusion_matrix_train
roc.curve(training_data$TARGET, training_data$predict_score_cart[,2])


pred.test_cart_model <- predict(cart_model, newdata = test_data[,1:15])

test_data$predict_class_cart <- predict(cart_model, test_data[,1:15], type = "class") 

test_data$predict_prob_cart <- predict(cart_model, test_data[,1:15], type = "prob") 
confusion_matrix_test_cart <-confusionMatrix(test_data$TARGET, test_data$predict_class_cart)

confusion_matrix_test_cart

accuracy.meas(test_data$TARGET, pred.test_cart_model[,2])

roc.curve(test_data$TARGET, pred.test_cart_model[,2])


# End of CART without and modification in smpling 




#CART_over sampling method 
set.seed(seed_no)


training_data <- ovun.sample(TARGET ~., data = training_data[,1:15], method = "over")$data
table(training_data$TARGET)


rpart.ctrl_over <- rpart.control(minsplit = 200, minbucket = 200, cp =0.00, xval = 50)
tree_over  <- rpart(TARGET~., data = training_data[,1:15], method = "class", control = rpart.ctrl_over) 
tree_over

cp_over <- tree_over$cptable[which.min(tree_over$cptable[,"xerror"]),"CP"]
cp_over

printcp(tree_over)

plotcp(tree_over)

varImp(tree_over)

tree_over <- prune(tree_over, cp = 0.004)
fancyRpartPlot(tree_over)
training_data$predict_class_over <- predict(tree_over, training_data[,1:15] , type = "class")
training_data$predict_prob_over <- predict(tree_over, training_data[,1:15] , type = "prob")
confusion_matrix_over <- confusionMatrix(training_data$TARGET, training_data$predict_class_over)
confusion_matrix_over
roc.curve(training_data$TARGET, training_data$predict_prob_over[,2])


pred.tree_over <- predict(tree_over, newdata = test_data[,1:15])
test_data$predict_class_cart_over <- predict(tree_over, test_data[,1:15], type = "class")
test_data$predict_prob_cart_over <- predict(tree_over, test_data[,1:15], type = "prob")
confusion_matrix_test_over <-confusionMatrix(test_data$TARGET, test_data$predict_class_cart_over)
confusion_matrix_test_over
roc.curve(test_data$TARGET, pred.tree_over[,2])
#ENd of CART_over sampling method 



#under sampling for CART model
set.seed(seed_no)
training_data <- ovun.sample(TARGET ~., data = training_data[,1:15], method = "under")$data
table(training_data$TARGET)


rpart.ctrl_under <- rpart.control(minsplit = 200, minbucket = 100, cp =0.00, xval = 10)
tree_under  <- rpart(TARGET~., data = training_data[,1:15], method = "class", control = rpart.ctrl_under) 

cp_under <- tree_under$cptable[which.min(tree_under$cptable[,"xerror"]),"CP"]

cp_under

printcp(tree_under)
plotcp(tree_under)
tree_under <- prune(tree_under, cp = cp_under)
fancyRpartPlot(tree_under)

training_data$predict_class_under <- predict(tree_under, training_data[,1:15] , type = "class")
training_data$predict_prob_under <- predict(tree_under, training_data[,1:15] , type = "prob")
confusion_matrix_under <- confusionMatrix(training_data$TARGET, training_data$predict_class_under)

confusion_matrix_under


pred.tree_under <- predict(tree_under, newdata = test_data[,1:15])
test_data$predict_class_CART_under <- predict(tree_under, newdata = test_data[,1:15], type = "class")
test_data$predict_prob_CART_under <- predict(tree_under, newdata = test_data[,1:15], type = "prob")
confusion_Matrix_test_under <- confusionMatrix(test_data$TARGET, test_data$predict_class_CART_under)
confusion_Matrix_test_under
roc.curve(test_data$TARGET, pred.tree_under[,2])



#using both under and over sampling for CART model 
set.seed(seed_no)
training_data<- ovun.sample(TARGET~., data = training_data[,1:15], method = "both", p=0.5)$data
table(training_data$TARGET)

rpart.ctrl_both <- rpart.control(minsplit = 200, minbucket =100, cp =0.00, xval = 50)

tree_balanced_both  <- rpart(TARGET~., data = training_data[,1:15], method = "class", control = rpart.ctrl_both) 
#tree_balanced_both
cp_both <- tree_balanced_both$cptable[which.min(tree_balanced_both$cptable[,"xerror"]),"CP"]
cp_both

printcp(tree_balanced_both)
plotcp(tree_balanced_both)

tree_balanced_both <- prune(tree_balanced_both, cp = cp_both)
fancyRpartPlot(tree_balanced_both)
training_data$predict_class_both <- predict(tree_balanced_both, training_data[, 1:15] , type = "class")
training_data$predict_prob_both <- predict(tree_balanced_both, training_data[, 1:15] , type = "prob")
confusion_matrix_both <- confusionMatrix(training_data$TARGET, training_data$predict_class_both)
confusion_matrix_both

pred.tree_test_balanced <- predict(tree_balanced_both, newdata = test_data[,1:15])
test_data$predict_class_CART_both <- predict(tree_balanced_both, newdata = test_data[,1:15], type = "class")

test_data$predict_prob_CART_both <- predict(tree_balanced_both, newdata = test_data[,1:15], type = "prob")
confusion_matrix_test_both <- confusionMatrix(test_data$TARGET, test_data$predict_class_CART_both)
  
confusion_matrix_test_both

accuracy.meas(test_data$TARGET, pred.tree_test_balanced[,2])
roc.curve(test_data$TARGET, pred.tree_test_balanced[,2])




 #rose model for CART
set.seed(seed_no)
training_data <- ROSE(TARGET~., data = training_data[,1:15], seed = 1)$data
table(training_data$TARGET)
rpart.ctrl_rose <- rpart.control(minsplit = 200, minbucket =100, cp =0.00, xval = 10)
tree_rose <- rpart(TARGET~., data = training_data[,1:15], method = "class", control = rpart.ctrl_rose)
cp_rose <- tree_rose$cptable[which.min(tree_rose$cptable[,"xerror"]),"CP"]
cp_rose
printcp(tree_rose)
plotcp(tree_rose)
 
tree_rose <- prune(tree_rose, cp = cp_rose)
fancyRpartPlot(tree_rose)

training_data$predict_class_rose <- predict(tree_rose, training_data[, 1:15] , type = "class")
confuion_matrix_rose<- confusionMatrix(training_data$TARGET, training_data$predict_class_rose)
confuion_matrix_rose
pred.tree_test_rose <- predict(tree_rose, newdata = test_data[,1:15])

test_data$predict_class_CART_rose <- predict(tree_rose, newdata = test_data[,1:15], type = "class")
test_data$predict_prob_CART_rose <- predict(tree_rose, newdata = test_data[,1:15], type = "prob")

confusion_matrix_test_rose <- confusionMatrix(test_data$TARGET, test_data$predict_class_CART_rose)
confusion_matrix_test_rose
accuracy.meas(test_data$TARGET, pred.tree_test_rose[,2])
roc.curve(test_data$TARGET, pred.tree_test_rose[,2])


#random forest 
set.seed(seed_no)

#install.packages("randomForest")
library(randomForest)

rndFor = randomForest(TARGET ~ ., data = training_data[1:15],
                      ntree=81, mtry = 14, nodesize = 5,
                      importance=TRUE)

print(rndFor)

rndFor$err.rate
plot(rndFor, main="")
legend("topright", c("OOB", "0", "1"), text.col=1:6, lty=1:3, col=1:3)
title(main="Error Rates Random Forest")
print(rndFor$importance)

set.seed(seed_no)

tRndFor = tuneRF(x = training_data[,2:15],
                 y=training_data$TARGET,
                 mtryStart = 3,
                 ntreeTry = 51,
                 stepFactor = 1.5,
                 improve = 0.0001,
                 trace=TRUE,
                 plot = TRUE,
                 doBest = TRUE,
                 nodesize = 5,
                 importance=TRUE
)
#importance(tRndFor)


training_data$predict.class = predict(tRndFor, training_data[,1:15], type="class")
training_data$prob1_rf = predict(tRndFor, training_data[,1:15], type="prob")[,"1"]

confusionMatrix(training_data$TARGET, training_data$predict.class)

qs=quantile(training_data$prob1_rf,prob = seq(0,1,length=11))
print(qs)

print(qs[10])
threshold=0.5

mean((training_data$TARGET[training_data$prob1_rf>threshold])=="1")

#TEST data
test_data$predict.class_rf = predict(tRndFor, test_data[,1:15], type="class")
test_data$prob1_rf = predict(tRndFor, test_data[,1:15], type="prob")

confusion_matrix_test_rf <- confusionMatrix(test_data$TARGET, test_data$predict.class_rf)
confusion_matrix_test_rf


# #cross validation method for random forest 
set.seed(seed_no)  
 control <- trainControl( method="repeatedcv", number=5, repeats=3)
 metric <- "Accuracy"
 set.seed(seed_no)
 mtry <- sqrt(ncol(training_data[,1:15]))
 tunegrid <- expand.grid(.mtry=mtry)
 
 rf_default <- train(TARGET~., data=training_data[,1:15], method="rf", metric=metric, tuneLength = 5,  trControl=control)
 print(rf_default)
 plot(rf_default)
 rf_default$results
 best_mtry<- rf_default$bestTune
 control_up <- trainControl( method="repeatedcv", number=5, repeats = 3)
 metric <- "Accuracy"
 set.seed(seed_no)
 mtry <- best_mtry #sqrt(ncol(training_data[,1:15]))
 tunegrid <- expand.grid(.mtry=mtry)
 rf_default_up <- train(TARGET~., data=training_data[,1:15], method="rf", metric=metric, tuneGrid=tunegrid, trControl=control_up)
 print(rf_default_up)
 rf_default_up$results
 predict_class_rf <- predict(rf_default_up, training_data[,1:15])
 training_data$predict_class_rf <- predict(rf_default_up, training_data[,1:15], type = "raw")
 confusion_Matrix_rf <- confusionMatrix(training_data$TARGET, training_data$predict_class_rf)
 confusion_Matrix_rf
 
 test_data$predict_class_rf <- predict(rf_default_up, test_data[,1:15], type = "raw")
 test_data$predict_prob_rf <- predict(rf_default_up, test_data[,1:15], type = "prob")
confusion_matrix_test_rf_up <- confusionMatrix(test_data$TARGET, test_data$predict_class_rf)
confusion_matrix_test_rf_up

#https://www.analyticsvidhya.com/blog/2017/02/introduction-to-ensembling-along-with-implementation-in-r/

confusion_matrix_train
confusion_matrix_test_cart
confusion_matrix_test_over
confusion_matrix_over
confusion_Matrix_test_under$table
confusion_matrix_test_both
confusion_matrix_test_rose$table
confusion_matrix_test_rf$table
confusion_matrix_test_rf_up

 
test_data$pred_avg <- (test_data$prob1_rf + test_data$predict_prob_CART_both)/2

#Splitting into binary classes at 0.5
test_data$pred_avg <- as.factor(ifelse(test_data$pred_avg[,1] > 0.5,'1','0'))
confusionMatrix(test_data$TARGET, test_data$pred_avg)

test_data[, test_data$pred_avg == "1"]
final <- subset(test_data, test_data$pred_avg == "1") #test_data[test_data$pred_avg == "1"]
describe(final[,1:15])
test_data[test_data$pred_avg]
range(final$LEN_OF_RLTN_IN_MNTH)

