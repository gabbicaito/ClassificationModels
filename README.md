# Classification Models

```{r, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, warning=FALSE, message=FALSE)
```

```{r, warning=FALSE, message=FALSE}
library(tidyverse)
```

## Curse of Dimensionality

When the number of input variables $p$ is large, there tends to be a deterioration in the performance of k-nearest neighbors (KNN) and other _local_ approaches that perform prediction using only observations that are _near_ the test observation for which a prediction will be made. This phenomenon is known as the __curse of dimensionality__, and is tied into the fact that non-parametric approaches often perform poorly when $p$ is large.

Suppose that we have a set of observations, each with measurements on $p=1$ feature, $X$. Assume that $X$ is _uniformly_, or evenly, distributed on $[0, 1]$. We plan to predict a test observation's response using only observations that are within 0.05 units of that test observation. For example, in order to predict the response for a test observation with $X = 0.60$, we'll use observations in the range $[0.55, 0.65]$.

If $x\in [0.05,0.95]$, then the observations we will use are in $[x−0.05,x+0.05]$ and thus represents a length of 0.1. If $x<0.05$, we will use observations in $[0,x+0.05]$, which represents a fraction of $(100x+5)$%. Similarly, if $x>0.95$, then the fraction of observations we will use is $(105−100x)$%. To find the average fraction of the available observations that we will use to make the prediction, we evaluate $\int_{0.05}^{0.95} 10 \, dx + \int_{0}^{0.05} (100x+5) \, dx + \int_{0.95}^{1} (105-100x) \, dx$, which equals $9.75$. So we can conclude that the fraction of available observations we will use to make the prediction is $9.75$%, on average.

Now suppose we have a set of observations, each with measurements on $p=2$ features, $X_1$ and $X_2$. Assume that both inputs are uniformly distributed on $[0, 1] \times [0, 1]$. Again, we'll predict using only observations that are within 0.05 units of $X_1$ _and_ 0.05 units of $X_2$. 

If we assume that $X_1$ and $X_2$ are independent, the fraction of available observations we will use to make the prediction is $9.75$% $\times 9.75$%$=0.950625$%.

Likewise, if $p=100$, we can conclude that the fraction of available observations we will use to make the prediction is $9.75$%$^{100} \approx 0$%.

## Classification Method Analysis

Suppose we take a data set, divide it into a 50-50% training-testing split, and try out two different classification procedures. Logistic regression gets an error rate of 20% on the training data and 30% on the testing data. 1-nearest neighbors ($k=1$) gets an average error rate (averaged over both testing and training data sets) of 18%. 

If we use KNN with $k=1$ for classification of new observations, we have a training error rate of $0$%, because in this case we have $P(Y=j|X=x_i)=I(y_i=j)$, which equals $1$ if $y_i=j$ and $0$ if not. We do not make any errors on the training data within this setting, which explains the $0$% training error rate. However, we have an average error rate of $18$%, which implies a test error rate of $36$% for KNN, which is greater than the test error rate for logistic regression of $30$%. So, we should prefer to use logistic regression for classification of new observations because of its lower test error rate.

## Question 3

The `BlueJays` data set represents measurements on nine variables for a sample of 123 blue jays captured near Oberlin College. For birds that were captured multiple times, the values were averaged. 

Variable|Description
-----|-------
`BirdID`|	ID tag for bird
`KnownSex`|	Sex coded as F or M
`BillDepth`|	Thickness of the bill measured at the nostril (in mm)
`BillWidth`	|Width of the bill (in mm)
`BillLength`|	Length of the bill (in mm)
`Head`	|Distance from tip of bill to back of head (in mm)
`Mass`	|Body mass (in grams)
`Skull`	|Distance from base of bill to back of skull (in mm)
`Sex`	|Sex coded as 0=female or 1=male

In R, we can access this data set from the `Stat2Data` library. 

```{r}
library(Stat2Data)
data(BlueJays)
```

For this data set, my machine learning models to predict sex based on body measurement should only have _six_ input variables. There are 8 total variables, but one of those variables is the response variable, sex, and another one is BirdID, which is just the ID tag for each individual bird and does not play a role in predicting sex.

I will create a testing and training data set, using a 70-30% split. 

```{r}
library(caret)
set.seed(366)
trainIndex <- createDataPartition(BlueJays$Sex, p=0.7, list=FALSE, times=1)
train <- BlueJays[trainIndex,]
nrow(train)
test <- BlueJays[-trainIndex,]
nrow(test)
```

There are 87 observations in the training data set and 36 observations in the testing data set.

Now I will fit a linear discriminant model to predict sex based on body measurements of the blue jays in my training data. 

```{r}
library(caret)
model_lda<-train(form=KnownSex~BillDepth+BillWidth+BillLength+Head+Mass+Skull, data=train, method="lda2")
model_lda
model_lda$finalModel
```

The final model is $KS=1.9436686BD-0.2588895BW-4.3511417BL+4.9377406H-0.0881729M-4.0584771S$. Bill length, head, skull, and maybe bill depth appear to be good discriminants, since their coefficients all have relatively large absolute values. This model has an accuracy of 76.56% and a kappa of 52.74%.

I will now build a confusion matrix for the linear discriminant model in order to comment on the performance of the model.

```{r}
confusionMatrix(data=predict(model_lda, test), reference=test$KnownSex)
```

This model incorrectly predicted male blue jays as female 0% of the time and it incorrectly predicted female blue jays as male 8.33% of the time. That is pretty good!

I will plot the decision boundary for my LDA model using bill length and body mass as my "axis" variables and setting all other input variables to their _mean_ values. 

```{r}
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
```

Based on this plot, my LDA model does not do a very good job classifying blue jay sex correctly. In fact, it appears as though it does a better job classifying blue jay sex incorrectly than it does correctly, as can be seen by the fact that the majority of female "points," or known females, fall within the male prediction boundary and vice versa.

Next I will fit a quadratic discriminant model to predict sex based on body measurements of the blue jays in my training data.

```{r}
model_qda<-train(form=KnownSex~BillDepth+BillWidth+BillLength+Head+Mass+Skull, data=train, method="qda")
model_qda
model_qda$finalModel
```

```{r}
cor(BlueJays$Skull,BlueJays$Head)
```

Without getting into what this means mathematically, the rank deficiency error I am getting indicates that my model is too complex. I will fix this by building a model that doesn't include skull to "simplify" the model a bit, since skull and head are highly correlated, as indicated by the correlation coefficient of 0.668.

```{r}
model_qda2<-train(form=KnownSex~BillDepth+BillWidth+BillLength+Head+Mass, data=train, method="qda")
model_qda2
model_qda2$finalModel
```

My QDA model has an accuracy of 69.71% and a kappa of 39.73%, which isn't bad. However, this means that this model is slightly less accurate than my LDA model. 

I will now build a confusion matrix for the quadratic discriminant model so that I can comment on the performance of the model.

```{r}
confusionMatrix(data=predict(model_qda2, test), reference=test$KnownSex)
```

This model incorrectly predicted male blue jays as female 0% of the time and it incorrectly predicted female blue jays as male 8.33% of the time. Based on this, it seems to be performing similarly to the LDA model.

Now I will plot the decision boundary for my QDA model using bill length and body mass as my "axis" variables and setting all other input variables to their _mean_ values. 

```{r}
grid_mean_qda <- grid %>% mutate(BillDepth = mean(train$BillDepth),
                        BillWidth = mean(train$BillWidth),
                        Head = mean(train$Head),
                        Skull = mean(train$Skull))

qda_predictions <- predict(model_qda2,grid_mean_qda)

grid_mean_qda %>% ggplot(aes(x=BillLength, y=Mass)) + geom_tile(aes(fill=qda_predictions), alpha=0.3) + geom_point(data=train, aes(x=BillLength, y=Mass, col=KnownSex))
```

This plot indicates that the QDA model is classifying smaller bill lengths and smaller masses, as well as larger bill lengths and larger masses, as female, while it is classifying smaller bill lengths and larger masses, as well as larger bill lengths and smaller masses, as male. Looking at the actual data points within this decision boundary, however, indicates that the model's decision boundary does not seem to be accurately placed, i.e., the male data points are scattered within both the male and the female boundaries, and the same is true for the female data points.  

Next I'll plot the decision boundary for my QDA model, using bill length and body mass as my "axis" variables and setting all other input variables to their _maximum_ values. 

```{r}
grid_max_qda <- grid %>% mutate(BillDepth = max(train$BillDepth),
                        BillWidth = max(train$BillWidth),
                        Head = max(train$Head),
                        Skull = max(train$Skull))

qda_predictions <- predict(model_qda2,grid_max_qda)

grid_max_qda %>% ggplot(aes(x=BillLength, y=Mass)) + geom_tile(aes(fill=qda_predictions), alpha=0.3) + geom_point(data=train, aes(x=BillLength, y=Mass, col=KnownSex))
```

When I do this, the female decision boundary covers the bottom half of the grid, rather than covering both the very top and the very bottom. In context, this means that, when all other variables are set to their maximum value, the QDA model is classifying blue jays with smaller masses as female, in general, though obviously this varies a bit depending on bill length. On the contrary, the male decision boundary covers the top half of the grid rather than being confined to the interior of the grid as was the case when all other variables were set to their maximum values. I context, this meas that, in general, the QDA models seems to be classifying blue jays with larger masses as male, again keeping in mind that this decision boundary varies a bit with bill length. 

Next I will fit a k-nearest neighbors model to predict sex based on body measurements of the blue jays in my training data, using at least a variety of different values of k in my model building.

```{r}
k <- data.frame(k=5:50)
model_knn = train(form=KnownSex~Skull+Head+Mass+BillLength+BillDepth+BillWidth, data=train, method="knn", tuneGrid=k)
model_knn
```

For my model, the "optimal" k is k=15. I know this because, out of the 46 k values used in my model building, the one with the highest accuracy was k=15. The accuracy of the model with k=15 is 70.46%, which is decently high This means that this k value optimizes the accuracy of my model. This can also be seen in the plot of the model below. The "peak" accuracy is at k=15.

```{r}
plot(model_knn)
```

Now I will build a confusion matrix for the k-nearest neighbor model (using optimal k) in order to comment on the performance of the model.

```{r}
confusionMatrix(data=predict(model_knn, test), reference=test$KnownSex)
```

Based on my confusion matrix, this model is correctly classifying female blue jays 38.89% of the time and male blue jays 44.44% of the time. Meanwhile, it is incorrectly classifying female blue jays as male 11.11% of the time and male blue jays as female 5.56% of the time. This is not bad, since the model seems to be correctly predicting the sex of blue jays the majority of the time. However, based on this, this model is not performing as well as the previous two models.

```{r}
grid_knn <- grid %>% mutate(BillDepth = mean(train$BillDepth),
                        BillWidth = mean(train$BillWidth),
                        Head = mean(train$Head),
                        Skull = mean(train$Skull))

knn_predictions <- predict(model_knn,grid_knn)

grid_knn %>% ggplot(aes(x=BillLength, y=Mass)) + geom_tile(aes(fill=knn_predictions), alpha=0.3) + geom_point(data=train, aes(x=BillLength, y=Mass, col=KnownSex))
```

As you can see above, it doesn't quite make sense to plot a decision boundary for this model for k-nearest neighbors, because the model is too complex, as it the decisin boundary, and it doesn't really add much value to our analysis because of that.

Overall, I think the model that is "best" for predicting sex based on body measurements is the LDA model. This model has the highest accuracy and kappa, implying that it is the model that can predict blue jay sex most accurately out of the three models I built. However, I do have some hesitations about this model. First off, though it has the highest accuracy of the three models I built, its accuracy still isn't very high at ~70%. I definitely wouldn't use this model in the real world if I needed a model that accurately predicted blue jay sex. Additionally, the decision boundary I plotted for this model gives me some concerns. Based on the distribution of the data and the two different sex classes within the data, a linear decision boundary really doesn't give an accurate picture of what is going on in the data. So, though this model is the best out of the three accuracy-wise, it is not good enough for practical use. 
