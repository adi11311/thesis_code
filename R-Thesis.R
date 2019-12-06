setwd("C:/Users/aivko/Desktop/R_Thesis")
install.packages("anytime")
install.packages("keras")
install.packages("tensorflow")
install_tensorflow()
install.packages("tsfknn")
install.packages("forecast")
library(anytime)
library(dplyr)
library(ggplot2)
library(keras)
library(tensorflow)
library(tsfknn)
library(forecast)
library(lmtest)

#loading the ABB data set and adjusting the timestamp
for_sending <- read.csv("for_sending.csv")
for_sending$event_timestamp <- as.POSIXct(strptime(for_sending$event_timestamp, "%Y-%m-%d %H:%M:%S"))

#adding an error_ratio column
count_the_robots <- count(for_sending, an_robot)
count_the_severity <- count(for_sending, message_severity, an_robot)
only_errors <- filter(count_the_severity, message_severity == "Error")
only_errors <- rename(only_errors, Number.of.errors = n)
merge_errors_robots <- left_join(only_errors, count_the_robots)
error_ratios <- mutate(merge_errors_robots, error_ratio = Number.of.errors/n)

#making a top 3 of the robots with the highest error ratios
target_robots <- c("robot_138", "robot_91", "robot_137")
top_3_highest_error_robots <- filter(for_sending, an_robot %in% target_robots)
top_3_highest_error_robots <- arrange(top_3_highest_error_robots, an_robot)

#adding case id's for every day a robot is active
top_3_highest_error_robots$case_id <- as.Date(top_3_highest_error_robots$event_timestamp)
top_3_highest_error_robots$case_id <- as.numeric(top_3_highest_error_robots$case_id)

#selecting robot_138
robot_138 <- filter(top_3_highest_error_robots, an_robot == "robot_138")
write.csv(robot_138, file = "robot_138.csv")

#selecting robot_91
robot_91 <- filter(top_3_highest_error_robots, an_robot == "robot_91")
write.csv(robot_138, file = "robot_91.csv")

#selecting robot_137
robot_137 <- filter(top_3_highest_error_robots, an_robot == "robot_91")
write.csv(robot_138, file = "robot_137.csv")

#importing the extracted robot_138 CSV files from Disco (Process mining)
new_robot138 <- read.csv("new_robot_138.csv", sep=";")
new_robot138 <- new_robot138[-5:-6]
names(new_robot138)[1] <- "case_id"
names(new_robot138)[2] <- "message_severity"
names(new_robot138)[3] <- "an_title"
names(new_robot138)[4] <- "event_timestamp"
new_robot138$event_timestamp <- gsub('.000', '', new_robot138$event_timestamp)
new_robot138$event_timestamp <- as.POSIXct(strptime(new_robot138$event_timestamp, "%Y/%m/%d %H:%M:%S"))
new_robot138 <- new_robot138 %>% 
  select(an_title, everything())
new_robot138 <- new_robot138[!(new_robot138$an_title==""),]

#importing the extracted robot_91 CSV files from Disco (Process mining)
new_robot91 <- read.csv("new_robot_91.csv", sep=";")
new_robot91 <- new_robot91[-5:-6]
names(new_robot91)[1] <- "case_id"
names(new_robot91)[2] <- "message_severity"
names(new_robot91)[3] <- "an_title"
names(new_robot91)[4] <- "event_timestamp"
new_robot91$event_timestamp <- gsub('.000', '', new_robot91$event_timestamp)
new_robot91$event_timestamp <- as.POSIXct(strptime(new_robot91$event_timestamp, "%Y/%m/%d %H:%M:%S"))
new_robot91 <- new_robot91 %>% 
  select(an_title, everything())
new_robot91 <- new_robot91[!(new_robot91$an_title==""),]

#importing the extracted robot_137 CSV files from Disco (Process mining)
new_robot137 <- read.csv("new_robot_137.csv", sep=";")
new_robot137 <- new_robot137[-5:-6]
names(new_robot137)[1] <- "case_id"
names(new_robot137)[2] <- "message_severity"
names(new_robot137)[3] <- "an_title"
names(new_robot137)[4] <- "event_timestamp"
new_robot137$event_timestamp <- gsub('.000', '', new_robot137$event_timestamp)
new_robot137$event_timestamp <- as.POSIXct(strptime(new_robot137$event_timestamp, "%Y/%m/%d %H:%M:%S"))
new_robot137 <- new_robot137 %>% 
  select(an_title, everything())
new_robot137 <- new_robot137[!(new_robot137$an_title==""),]

#the plot for robot_138 showing 
total_plot <- ggplot(new_robot138, aes(event_timestamp, an_title, color = message_severity)) +
  geom_point() +
  ggtitle("an_title's for robot_138 categorized per message_severity")
total_plot

##predictions for robot_138##

#experiment for robot_138, creating the (train) data set
data <- data.matrix(new_robot138)
data <- data[,1:3]
data <- data[,-2]
train_data <- data[1:5563,]

#normalizing the data set
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)

#generator function for robot_138
generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 3) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size-1, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows),
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]]-1,
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }           
    list(samples, targets)
  }
}

#setting up the lookback, step, delay and batch_size
lookback <- 800
step <- 3
delay <- 80
batch_size <- 128

#train, val and test sets
train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 5563,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)
val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 5564,
  max_index = 7418,
  step = step,
  batch_size = batch_size
)

test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 7419,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)

#how many steps to draw from val_gen in order to see the entire validation set
val_steps <- (7418 - 5564 - lookback) / batch_size

#how many steps to draw from test_gen in order to see the entire test set
test_steps <- (nrow(data) - 7419 - lookback) / batch_size

#view the normalized data plot
data_norm <- as.data.frame(data)
data_norm$message_severity <- as.factor(data_norm$message_severity)
data_norm$message_severity <- as.numeric(data_norm$message_severity)
data_norm$message_severity <- as.factor(data_norm$message_severity)
data_norm$message_severity <- as.numeric(data_norm$message_severity)
data_norm$message_severity[data_norm$message_severity == 1]  <- "Error" 
data_norm$message_severity[data_norm$message_severity == 2]  <- "Information" 
data_norm$message_severity[data_norm$message_severity == 3]  <- "Warning" 
dataplot <- ggplot(data_norm, aes(x = 1:nrow(data), an_title, color = message_severity)) +
  geom_point() +
  ggtitle("an_title's for robot_138 categorized per message_severity (normalized)")

dataplot

#baseline non-machine learning prediction model
evaluate_naive_method <- function() {
  batch_maes <- c()
  for (step in 1:test_steps) {
    c(samples, targets) %<-% test_gen()
    preds <- samples[,dim(samples)[[2]],2]
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}

evaluate_naive_method()

#predict the denormalized mae's
mae_pred <- evaluate_naive_method() * std
mae_pred

#keras model
model <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]]), activation = "relu") %>% 
  layer_dense(units = 1)

#adjust loss to mse in order to calculate mse's
model %>% compile(
  optimizer = optimizer_adam(clipnorm = 1.),
  loss = "mean_squared_error"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 10,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)

history_df <- as.data.frame(history)
ggplot(history_df, aes(x = epoch, y = value, color = data)) +
  geom_line() +
  geom_point() +
  ggtitle("RNN model history for robot_138")

#save the model as a hdf5 file 
save_model_hdf5(model, "model_138_mae.h5")

#after changing the model to mse, save the mse version as an hdf5 file
save_model_hdf5(model, "model_138_mse.h5")

#calculate the loss for the model (mae by default, change to "mse" in the model if wanted)
loss <- model %>% evaluate_generator(
  test_gen, test_steps)

#change to mse when calculating mse
mae_138 <- loss * std
mae_138

######predictions for robot_138######

#adjust the generator function to obtain the mae/mse
generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 3) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size-1, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows),
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]]-1,
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }           
    list(samples)
  }
}

preds <- model %>% predict_generator(test_gen, steps = test_steps)
denorm_pred <- preds * std + mean
denorm_pred <- as.data.frame(denorm_pred)
names(denorm_pred)[1] <- "an_title"
denorm_pred$message_severity <- 0
names(denorm_pred)[2] <- "message_severity"
denorm_pred$message_severity <- ifelse(denorm_pred$an_title < 4, 1, 2)
data_for_pred <- data.matrix(new_robot138)
data_for_pred <- data_for_pred[,1:3]
data_for_pred <- data_for_pred[,-2]
forecasted_values <- rbind(data_for_pred,denorm_pred)
forecasted_values["message_severity"] <- if_else(forecasted_values$message_severity == 1, "Error", "Information")

ggplot(forecasted_values, aes(x = 1:nrow(forecasted_values), y = an_title, color = message_severity)) +
  geom_point() +
  geom_vline(xintercept = 9272, color = "red") +
  ggtitle("RNN prediction for robot_138")

#experiment for robot_91, creating the (train) data set
data91 <- data.matrix(new_robot91)
data91 <- data91[,1:3]
data91 <- data91[,-2]
train_data91 <- data91[1:5977,]

#normalizing the data set
mean91 <- apply(train_data91, 2, mean)
std91 <- apply(train_data91, 2, sd)
data91 <- scale(data91, center = mean, scale = std)

#generator function for robot_91
generator <- function(data91, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 3) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size-1, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows),
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]]-1,
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }           
    list(samples, targets)
  }
}

#train, val and test sets
train_gen91 <- generator(
  data91,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 5977,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)

val_gen91 = generator(
  data91,
  lookback = lookback,
  delay = delay,
  min_index = 5978,
  max_index = 7970,
  step = step,
  batch_size = batch_size
)

test_gen91 <- generator(
  data91,
  lookback = lookback,
  delay = delay,
  min_index = 7971,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)

#how many steps to draw from val_gen in order to see the entire validation set
val_steps91 <- (7970 - 5978 - lookback) / batch_size

#how many steps to draw from test_gen in order to see the entire test set
test_steps91 <- (nrow(data91) - 7971 - lookback) / batch_size

#baseline non-machine learning prediction model
evaluate_naive_method91 <- function() {
  batch_maes <- c()
  for (step in 1:test_steps) {
    c(samples, targets) %<-% test_gen()
    preds <- samples[,dim(samples)[[2]],2]
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}

evaluate_naive_method91()

#predict the denormalized mae's
mae_pred91 <- evaluate_naive_method() * std91
mae_pred91 

#keras model
model91 <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]]), activation = "relu") %>% 
  layer_dense(units = 1)

#adjust loss to mse in order to calculate mse's
model91 %>% compile(
  optimizer = optimizer_adam(clipnorm = 1.),
  loss = "mae"
)

history91 <- model %>% fit_generator(
  train_gen91,
  steps_per_epoch = 10,
  epochs = 20,
  validation_data = val_gen91,
  validation_steps = val_steps91
)

plot(history91)

#save the model as a hdf5 file 
save_model_hdf5(model91, "model_91_mae.h5")

#after changing the model to mse, save the mse version as an hdf5 file
save_model_hdf5(model91, "model_91_mse.h5")

#calculate the loss for the model (mae by default, change to "mse" in the model if wanted)
loss91 <- model91 %>% evaluate_generator(
  test_gen91, test_steps91)

#change to mse when calculating mse
mae_91 <- loss91 * std91
mae_91

#experiment for robot_137, creating the (train) data set
data137 <- data.matrix(new_robot137)
data137 <- data137[,1:3]
data137 <- data137[,-2]
train_data137 <- data137[1:5977,]

#normalizing the data set
mean137 <- apply(train_data137, 2, mean)
std137 <- apply(train_data137, 2, sd)
data137 <- scale(data137, center = mean, scale = std)

#generator function for robot_137
generator <- function(data137, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 3) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size-1, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows),
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]]-1,
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }           
    list(samples, targets)
  }
}

#train, val and test sets
train_gen137 <- generator(
  data137,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 5977,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)

val_gen137 = generator(
  data137,
  lookback = lookback,
  delay = delay,
  min_index = 5978,
  max_index = 7970,
  step = step,
  batch_size = batch_size
)

test_gen137 <- generator(
  data137,
  lookback = lookback,
  delay = delay,
  min_index = 7971,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)

#how many steps to draw from val_gen in order to see the entire validation set
val_steps137 <- (7970 - 5978 - lookback) / batch_size

#how many steps to draw from test_gen in order to see the entire test set
test_steps137 <- (nrow(data91) - 7971 - lookback) / batch_size

#baseline non-machine learning prediction model
evaluate_naive_method137 <- function() {
  batch_maes <- c()
  for (step in 1:test_steps) {
    c(samples, targets) %<-% test_gen()
    preds <- samples[,dim(samples)[[2]],2]
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}

evaluate_naive_method137()

#predict the denormalized mae's
mae_pred137 <- evaluate_naive_method() * std
mae_pred137

#keras model
model137 <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]]), activation = "relu") %>% 
  layer_dense(units = 1)

#adjust loss to mse in order to calculate mse's
model137 %>% compile(
  optimizer = optimizer_adam(clipnorm = 1.),
  loss = "mae"
)

history137 <- model %>% fit_generator(
  train_gen137,
  steps_per_epoch = 10,
  epochs = 20,
  validation_data = val_gen91,
  validation_steps = val_steps91
)

plot(history137)

#save the model as a hdf5 file 
save_model_hdf5(model137, "model_137_mae.h5")

#after changing the model to mse, save the mse version as an hdf5 file
save_model_hdf5(model137, "model_137_mse.h5")

#calculate the loss for the model (mae by default, change to "mse" in the model if wanted)
loss137 <- model137 %>% evaluate_generator(
  test_gen137, test_steps137)

#change to mse when calculating mse
mae_137 <- loss137 * std
mae_137



######TSFKNN Predictions######



#robot_138
#tsfknn prediction for k = 1
data_ts <- ts(data_for_pred[0:9272,])
data_ts_df <- as.data.frame(data_ts)
data_ts_df <- data_ts_df[-2]
pred <- knn_forecasting(ts(data_ts_df), h = 1024, lags = 1:1024, k = 1, msas = "MIMO")

pred_df <- as.data.frame(pred[["prediction"]])
pred_df["message_severity"] <- if_else(pred_df$x == 1 | 
                                       pred_df$x == 8 |
                                       pred_df$x == 21 |
                                       pred_df$x == 24 |
                                       pred_df$x == 26, 1, 2)
names(pred_df)[1] <- "an_title"
with_extension <- rbind(data_for_pred, pred_df)
with_extension$message_severity[with_extension$message_severity == 1]  <- "Error" 
with_extension$message_severity[with_extension$message_severity == 2]  <- "Information" 
with_extension$message_severity[with_extension$message_severity == 3]  <- "Warning" 

ggplot(with_extension, aes(x = 1:nrow(with_extension), y = an_title, color = message_severity)) +
  geom_point() +
  geom_vline(xintercept = 9272, color = "red") + 
  ggtitle("A basic KNN prediction model with k = 1")

#tsfknn prediction for k = 2
pred <- knn_forecasting(ts(data_ts_df), h = 1024, lags = 1:1024, k = 2, msas = "MIMO")

pred_df <- as.data.frame(pred[["prediction"]])
pred_df["message_severity"] <- if_else(pred_df$x == 1 | 
                                         pred_df$x == 8 |
                                         pred_df$x == 21 |
                                         pred_df$x == 24 |
                                         pred_df$x == 26, 1, 2)
names(pred_df)[1] <- "an_title"
with_extension <- rbind(data_for_pred, pred_df)
with_extension$message_severity[with_extension$message_severity == 1]  <- "Error" 
with_extension$message_severity[with_extension$message_severity == 2]  <- "Information" 
with_extension$message_severity[with_extension$message_severity == 3]  <- "Warning" 

ggplot(with_extension, aes(x = 1:nrow(with_extension), y = an_title, color = message_severity)) +
  geom_point() +
  geom_vline(xintercept = 9272, color = "red") + 
  ggtitle("A basic KNN prediction model with k = 2")

#tsfknn prediction for k = 8
pred <- knn_forecasting(ts(data_ts_df), h = 1024, lags = 1:1024, k = 8, msas = "MIMO")

pred_df <- as.data.frame(pred[["prediction"]])
pred_df["message_severity"] <- if_else(pred_df$x == 1 | 
                                         pred_df$x == 8 |
                                         pred_df$x == 21 |
                                         pred_df$x == 24 |
                                         pred_df$x == 26, 1, 2)
names(pred_df)[1] <- "an_title"
with_extension <- rbind(data_for_pred, pred_df)
with_extension$message_severity[with_extension$message_severity == 1]  <- "Error" 
with_extension$message_severity[with_extension$message_severity == 2]  <- "Information" 
with_extension$message_severity[with_extension$message_severity == 3]  <- "Warning" 

ggplot(with_extension, aes(x = 1:nrow(with_extension), y = an_title, color = message_severity)) +
  geom_point() +
  geom_vline(xintercept = 9272, color = "red") + 
  ggtitle("A basic KNN prediction model with k = 8")

#calculating the metrics for robot_138
ro <- rolling_origin(pred, h = 1024)
KNN_mae_mse_rmse1 <- ro1$global_accu
KNN_mae_mse_rmse1

#robot_91
#tsfknn prediction for k = 1
data_for_pred91 <- data.matrix(new_robot91)
data_for_pred91 <- data_for_pred91[,1:3]
data_for_pred91 <- data_for_pred91[,-2]
data_for_pred91 <- as.data.frame(new_robot91)
data_ts91 <- ts(data_for_pred91[0:9961,])
data_ts_df91 <- as.data.frame(data_ts91)
data_ts_df91 <- data_ts_df91[-2]
data_ts_df91 <- data_ts_df91[-3]
data_ts_df91 <- data_ts_df91[-2]
pred91 <- knn_forecasting(ts(data_ts_df91), h = 1024, lags = 1:1024, k = 1)

ro91 <- rolling_origin(pred91, h = 1024)
KNN_mae_mse_rmse91 <- ro91$global_accu
KNN_mae_mse_rmse91

#robot_137
#tsfknn prediction for k = 1
data_for_pred137 <- as.data.frame(new_robot137)
data_ts137 <- ts(data_for_pred137[0:9961,])
data_ts_df137 <- as.data.frame(data_ts137)
data_ts_df137 <- data_ts_df137[-2]
data_ts_df137 <- data_ts_df137[-3]
data_ts_df137 <- data_ts_df137[-2]
pred137 <- knn_forecasting(ts(data_ts_df137), h = 1024, lags = 1:1024, k = 1)

ro137 <- rolling_origin(pred137, h = 1024)
KNN_mae_mse_rmse137 <- ro137$global_accu
KNN_mae_mse_rmse137



######ARIMA Predictions######



#robot_138 predictions

#the timeframe for robot_138 is from 03-08-2015 until 07-15-2015, which is 128 days
arima_predict <- forecast(auto.arima(ts(data_ts_df,frequency=128),D=1,stepwise = FALSE, approximation = FALSE), h=1024)
autoplot(arima_predict)
arima_predict1 <- as.data.frame(arima_predict)
arima_predict1 <- arima_predict1[1]
arima_predict1$ID <- seq.int(nrow(arima_predict1))
row.names(arima_predict1) <- arima_predict1$ID
names(arima_predict1)[1] <- "an_title"
arima_predict1 <- arima_predict1[-2]
arima_predict1 <- round(arima_predict1,0)
arima_predict1$message_severity <- 0
arima_predict1["message_severity"] <- if_else(arima_predict1$an_title == round(1) | 
                                                arima_predict1$an_title == round(8) |
                                                arima_predict1$an_title == round(21) |
                                                arima_predict1$an_title == round(24) |
                                                arima_predict1$an_title == round(26), 1, 2)
with_arima_ext <- rbind(data_for_pred, arima_predict1)
with_arima_ext$message_severity[with_arima_ext$message_severity == 1]  <- "Error" 
with_arima_ext$message_severity[with_arima_ext$message_severity == 2]  <- "Information" 
with_arima_ext$message_severity[with_arima_ext$message_severity == 3]  <- "Warning" 

ggplot(with_arima_ext, aes(x = 1:nrow(with_arima_ext), y = an_title, color = message_severity)) +
  geom_point() +
  geom_vline(xintercept = 9272, color = "red") + 
  ggtitle("An ARIMA prediction for robot_138")

accuracy(arima_predict)

#robot_91 predictions

#the timeframe for robot_91 is from 03-05-2015 until 05-17-2015, which is 73 days
arima_predict91 <- forecast(auto.arima(ts(data_ts_df91,frequency=73),D=1),h=1024)

accuracy(arima_predict91)

#robot_137 predictions

#the timeframe for robot_137 is from 03-08-2015 until 07-15-2015, which is 128 days
arima_predict137 <- forecast(auto.arima(ts(data_ts_df137,frequency=128),D=1),h=1024)

accuracy(arima_predict)
