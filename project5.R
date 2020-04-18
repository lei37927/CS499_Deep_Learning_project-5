install.packages("caret") 
install.packages("keras")
install.packages("tensorflow")

library(keras) # load keras
library(tensorflow)
install_tensorflow() # build a tensorflow environment
#install_keras()


library(caret)
library(pROC)
library(ggplot2)
library(rpart)
library(MASS)
library(Matrix)
library(tidyverse) # metapackage with lots of helpful functions
library(data.table)
library(ggthemes)



sta<- read.csv('spam.csv',head=T)
sta.x <- scale(sta[,-58]) 
sta.y<- sta[,58]
sta1 <- cbind.data.frame(sta.x,sta.y)

set.seed(888)

train_sub <- sample(nrow(sta1),0.8*nrow(sta1)) # random select
train_set <- data.matrix(sta1[train_sub,])
test_set <- data.matrix(sta1[-train_sub,]) ##


parameters <- cbind.data.frame(c(1:25),c(1:25),rep(0,25)) 
colnames(parameters) <- c('dropout','hiddenunits','accuracy')

set.seed(888)

for (i in 1:5){
  for(j in 1:5){
    
    
    model <- keras_model_sequential()
    
    model %>%
      
      layer_dense(units = 8+j, input_shape = 57) %>%
      
      layer_dropout(rate=0.2+0.1*i)%>% 
      
      layer_activation(activation = 'relu') %>%  
      
      layer_dense(units = 2) %>% 
      
      layer_activation(activation = 'sigmoid') 
    
    model %>% compile(
      loss = 'sparse_categorical_crossentropy', 
      optimizer = 'adam', 
      metrics = c('accuracy') 
    )
    model %>% fit(train_set[,-58], train_set[,58],epochs = 100, batch_size = 128,validation_split = 0.5) 
    
    loss_and_metrics <- model %>% evaluate(test_set[,-58], test_set[,58], batch_size = 128)
    
    
    parameters[5*(i-1)+j,1] <- i
    parameters[5*(i-1)+j,2] <- j
    parameters[5*(i-1)+j,3] <- loss_and_metrics$accuracy
    
    model %>% predict_classes(test_set[,-58])
   
    
  }
}


best_parameter_value <- parameters$dropout[parameters$accuracy==max(parameters[,3])]*0.1+0.2


best_parameter_value2 <- parameters$hiddenunits[parameters$accuracy==max(parameters[,3])]+8

### draw
set.seed(8889)
model <- keras_model_sequential()

model %>%
  
  layer_dense(units = best_parameter_value2, input_shape = 57) %>%
  
  layer_dropout(rate=best_parameter_value)%>% 
  
  layer_activation(activation = 'relu') %>%  
  
  layer_dense(units = 2) %>% 
  
  layer_activation(activation = 'sigmoid') 

model%>% compile(
  loss = 'sparse_categorical_crossentropy', 
  optimizer = 'adam', 
  metrics = c('accuracy') 
)
his <- model %>% fit(train_set[,-58], train_set[,58],epochs = 200, batch_size = 128,validation_split = 0.5) 

loss_and_metrics1<- model %>% evaluate(test_set[,-58], test_set[,58], batch_size = 128)

loss_and_metrics1$accuracy

a <- unlist(his$metrics)

val_loss <- cbind.data.frame(c(1:200),a[401:600])
colnames(val_loss) <- c('epochs','loss')
train_loss <- cbind.data.frame(c(1:200),a[1:200])
colnames(train_loss) <- c('epochs','loss')
train_val <- rbind.data.frame(val_loss,train_loss)
train_val$group <- '0'
train_val$group[1:200] <- 'val_loss'
train_val$group[201:400] <- 'train_loss'

plot <- ggplot(train_val, aes(x=epochs, y=loss, group=group,color = group)) + geom_line()+geom_point(size=2, shape=20)+geom_hline(yintercept = min(train_val[train_val$group=='val_loss',2]),linetype=3,colour='purple',size=2)
plot <- plot+ annotate("text", 150, 0.5, vjust = -1, label = paste0("min loss( purple line )  =  ",min(train_val[train_val$group=='val_loss',2])) )
plot+ annotate("text", 150, 0.4, vjust = -1, label = paste0("epochs = ",train_val$epochs[train_val$loss==min(train_val[train_val$group=='val_loss',2])]) )




### train with best parameters
set.seed(8889)
model <- keras_model_sequential()

model %>%
  
  layer_dense(units = best_parameter_value2, input_shape = 57) %>%
  
  layer_dropout(rate=best_parameter_value)%>% 
  
  layer_activation(activation = 'relu') %>%  
  
  layer_dense(units = 2) %>% 
  
  layer_activation(activation = 'sigmoid') 

model%>% compile(
  loss = 'sparse_categorical_crossentropy', 
  optimizer = 'adam', 
  metrics = c('accuracy') 
)
model %>% fit(train_set[,-58], train_set[,58],epochs = 200, batch_size = 128) 

loss_and_metrics2<- model %>% evaluate(test_set[,-58], test_set[,58], batch_size = 128)

loss_and_metrics2$accuracy##accuracy of test_set

model1 %>% predict_classes(test_set[,-58])

sum(train_set[,58])###the most frequent class is 0

dev.result <-abs(test_set[test_set[,58]=='0',58]- model %>% predict_classes(test_set[test_set[,58]=='0',-58]))
##921 
print(1-sum(dev.result)/(921-sum(test_set[,58])))##accuracy of calss 0


######## Extra Credits: 4-fold validation
sta.exe <- sta1

target = sta.exe$sta.y
sta.exe$sta.y = NULL
nrounds = 4
set.seed(1234)
folds = createFolds(factor(target), k = 4, list = FALSE)

test_acc <- cbind.data.frame(c(1:4),rep(0,4)) 
colnames(test_acc) <- c('k','accuracy')

for (this.round in 1:nrounds){      
  valid <- c(1:length(target)) [folds == this.round]
  dev <- c(1:length(target)) [folds != this.round]
  
  dtrain<-  data.matrix(sta.exe[dev,])
  
  dvalid <- data.matrix(sta.exe[valid,])
  
  ###
  
  model <- keras_model_sequential()
  
  model %>%
    
    layer_dense(units = best_parameter_value2, input_shape = 57) %>%
    
    layer_dropout(rate=best_parameter_value)%>% 
    
    layer_activation(activation = 'relu') %>%  
    
    layer_dense(units = 2) %>% 
    
    layer_activation(activation = 'sigmoid') 
  
  model%>% compile(
    loss = 'sparse_categorical_crossentropy', 
    optimizer = 'adam', 
    metrics = c('accuracy') 
  )
  model %>% fit(dtrain, target[dev],epochs = 200, batch_size = 128) 
  
  loss_and_metrics3<- model %>% evaluate(dvalid, target[valid], batch_size = 128)
  
  test_acc[this.round,2] <- loss_and_metrics3$accuracy
  
  
}

ggplot(test_acc, aes(x=k, y=accuracy)) + geom_line() + geom_point(size=4, shape=20)