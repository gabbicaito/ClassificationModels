library(tidyverse)

library(Stat2Data)
data(BlueJays)

library(caret)
set.seed(366)
trainIndex <- createDataPartition(BlueJays$Sex, p=0.7, list=FALSE, times=1)
train <- BlueJays[trainIndex,]
nrow(train)
test <- BlueJays[-trainIndex,]
nrow(test)

library(caret)
model_lda<-train(form=KnownSex~BillDepth+BillWidth+BillLength+Head+Mass+Skull, data=train, method="lda2")
model_lda
model_lda$finalModel

confusionMatrix(data=predict(model_lda, test), reference=test$KnownSex)

n_breaks <- 100
predA <- seq(min(train$BillLength), max(train$BillLength), length=n_breaks)
predB <- seq(min(train$Mass), max(train$Mass), length=n_breaks)
grid <- expand.grid(BillLength = predA, Mass = predB)
grid_lda <- grid %>% mutate(BillDepth = mean(train$BillDepth),
                            BillWidth = mean(train$BillWidth),
                            Head = mean(train$Head),
                            Skull = mean(train$Skull))
lda_predictions <- predict(model_lda,grid_lda)
grid_lda %>% ggplot(aes(x=BillLength, y=Mass)) + geom_tile(aes(fill=lda_predictions), alpha=0.3) + geom_point(data=train, aes(x=BillLength, y=Mass, col=KnownSex))

model_qda<-train(form=KnownSex~BillDepth+BillWidth+BillLength+Head+Mass+Skull, data=train, method="qda")
model_qda
model_qda$finalModel

cor(BlueJays$Skull,BlueJays$Head)

model_qda2<-train(form=KnownSex~BillDepth+BillWidth+BillLength+Head+Mass, data=train, method="qda")
model_qda2
model_qda2$finalModel

confusionMatrix(data=predict(model_qda2, test), reference=test$KnownSex)

grid_mean_qda <- grid %>% mutate(BillDepth = mean(train$BillDepth),
                                 BillWidth = mean(train$BillWidth),
                                 Head = mean(train$Head),
                                 Skull = mean(train$Skull))
qda_predictions <- predict(model_qda2,grid_mean_qda)
grid_mean_qda %>% ggplot(aes(x=BillLength, y=Mass)) + geom_tile(aes(fill=qda_predictions), alpha=0.3) + geom_point(data=train, aes(x=BillLength, y=Mass, col=KnownSex))

grid_max_qda <- grid %>% mutate(BillDepth = max(train$BillDepth),
                                BillWidth = max(train$BillWidth),
                                Head = max(train$Head),
                                Skull = max(train$Skull))
qda_predictions <- predict(model_qda2,grid_max_qda)
grid_max_qda %>% ggplot(aes(x=BillLength, y=Mass)) + geom_tile(aes(fill=qda_predictions), alpha=0.3) + geom_point(data=train, aes(x=BillLength, y=Mass, col=KnownSex))

k <- data.frame(k=5:50)
model_knn = train(form=KnownSex~Skull+Head+Mass+BillLength+BillDepth+BillWidth, data=train, method="knn", tuneGrid=k)
model_knn

plot(model_knn)

confusionMatrix(data=predict(model_knn, test), reference=test$KnownSex)

grid_knn <- grid %>% mutate(BillDepth = mean(train$BillDepth),
                            BillWidth = mean(train$BillWidth),
                            Head = mean(train$Head),
                            Skull = mean(train$Skull))
knn_predictions <- predict(model_knn,grid_knn)
grid_knn %>% ggplot(aes(x=BillLength, y=Mass)) + geom_tile(aes(fill=knn_predictions), alpha=0.3) + geom_point(data=train, aes(x=BillLength, y=Mass, col=KnownSex))
