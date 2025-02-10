### Statistical Learning Final Project
### Tyler Brown 01/1355033

# loading libraries
library(tidyverse)
library(haven)
library(fastDummies)
library(torch)

set.seed(2025)

# loading data
df_raw <- read_csv('1_data/House_Rent_Dataset.csv')

# descriptive plot on rent with numeric predictors
df_raw %>% select_if(is.numeric) %>% 
  pivot_longer(!Rent) %>% 
  ggplot(aes(x = value, y = Rent))+
  geom_point()+
  stat_summary_bin(color = 'red')+
  facet_wrap(~name, scales = 'free')

df_raw %>% 
  ggplot(aes(x = `Posted On`, y = log(Rent)))+
  stat_summary_bin()

df_raw %>% select(where(is.character), Rent) %>% 
  select(!c('Area Locality', 'Floor')) %>% 
  pivot_longer(!Rent) %>% 
  ggplot(aes(x = value, y = Rent))+
  stat_summary_bin(geom = 'bar', fill = 'red', alpha = 0.5)+
  stat_summary_bin()+
  facet_wrap(~name, scales = 'free', nrow = 1)+
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))+
  labs(x = '', y = 'Mean Rent', title = 'Mean rental price across predictors')
# ggsave('2_plots/mean_rental_cat_vars.png')

# justifying the log transformation
df_raw %>% 
  ggplot(aes(x = Rent))+
  geom_density()

df_raw %>% 
  ggplot(aes(x = log(Rent)))+
  geom_density()

# checking NAs
df_raw %>% 
  summarise_all(~sum(is.na(.))) %>% 
  t


# scaling numeric attributes
df <- df_raw %>%
  ungroup %>% 
  mutate(Rent = log(Rent)) %>% 
  mutate(`Posted On` = as.numeric(`Posted On`)) %>% 
  mutate_if(is.numeric, scale)


# recoding furnished as binary
df <- df %>% mutate(`Furnishing Status` = case_when(`Furnishing Status` == 'Unfurnished' ~ 0,
                                                    `Furnishing Status` == 'Semi-Furnished' ~ 1,
                                                    `Furnishing Status` == 'Furnished' ~ 1))


# making binary indicators across all categorical attributes
df <- dummy_cols(df, c('Floor', 
                       'Area Type', 
                       'Area Locality',
                       'City',
                       'Tenant Preferred', 
                       'Point of Contact'))

# removing initial categorical variables
df <- df %>% select(-c('Floor', 
                       'Area Type', 
                       'Area Locality', 
                       'City',
                       'Tenant Preferred', 
                       'Point of Contact'))

# renaming target variable for ease of use
df <- df %>% rename(target = Rent)


# constructing train/test split
train_indices <- sample(dim(df)[1], dim(df)[1] * 0.8)

df_train <- df[train_indices, ]
df_test <- df[-train_indices, ]

batch_size <- 32

# making tensor data
train_x <- torch_tensor(df_train %>% select(!target) %>% as.matrix)
train_y <- torch_tensor(df_train %>% select(target) %>% as.matrix)

test_x <- torch_tensor(df_test %>% select(!target) %>% as.matrix)
test_y <- torch_tensor(df_test %>% select(target) %>% as.matrix)

train_dl <- dataloader(tensor_dataset(train_x, train_y), 
                       batch_size = batch_size, 
                       shuffle = T)

test_dl <- dataloader(tensor_dataset(test_x, test_y), 
                      batch_size = batch_size, 
                      shuffle = F)



## ORGANIZING NEURAL NETWORK

# 2735 - 1 features

ModelClass <- nn_module(
  
  # initializing the model structure
  initialize = function() {
    # layer 1
    self$layer1 <- nn_linear(dim(df)[2] - 1, 4000)
    self$dropout1 <- nn_dropout(p = 0.25)
    
    # layer 2
    self$layer2 <- nn_linear(4000, 1000)
    self$dropout2 <- nn_dropout(p = 0.25)
    
    # layer 3
    self$layer3 <- nn_linear(1000, 100)
    self$dropout3 <- nn_dropout(p = 0.25)
    
    # output layer
    self$output <- nn_linear(100, 1)
    
  },
  
  # defining the forward feed function
  forward = function(x) {
    x %>% 
      self$layer1() %>% 
      nnf_relu() %>% 
      self$dropout1() %>% 
      self$layer2() %>% 
      nnf_relu() %>% 
      self$dropout2() %>% 
      self$layer3() %>% 
      nnf_relu() %>% 
      self$dropout3() %>% 
      self$output()
  }
)


# initializing hyperparameters

# also implementing early stopping in the event that the validation error is not improving over time
patience <- 5  


# tuning loop parameters

learning_rates <- c(0.001, 0.01, 0.1)
lambdas <- c(0.000001, 0.00001, 0.0001)

# number of epochs
num_epochs <- 20

# initializing list of models to be saved
model_list <- list()


for (learn_rate in learning_rates){
  for (lambda in lambdas){
    
    # defining model
    model <- ModelClass()
    
    optimizer <- optim_sgd(model$parameters, lr = learn_rate, weight_decay = lambda)
    loss_func <- nnf_mse_loss
    
    mean_train_losses <- c()
    mean_val_losses <- c()
    
    best_val_loss <- Inf
    counter <- 0
    
    for (epoch in 1:num_epochs) {
      
      model$train()
      train_losses <- c()  
      
      coro::loop(for (b in train_dl) {
        optimizer$zero_grad()
        output <- model(b[[1]])
        loss <- loss_func(output, b[[2]])
        loss$backward()
        optimizer$step()
        train_losses <- c(train_losses, loss$item())
      })
      
      model$eval()
      valid_losses <- c()
      
      coro::loop(for (b in test_dl) {
        output <- model(b[[1]])
        loss <- loss_func(output, b[[2]])
        valid_losses <- c(valid_losses, loss$item())
      })
      
      mean_train_loss <- mean(train_losses)
      mean_val_loss <- mean(valid_losses)
      
      mean_train_losses <- c(mean_train_losses, mean_train_loss)
      mean_val_losses <- c(mean_val_losses, mean_val_loss)
      
      model_list[[paste(learn_rate, '-', lambda)]] <- list(train_loss = mean_train_losses, val_loss = mean_val_losses)
      
      print(paste('Epoch: ', epoch, 'train loss:', mean_train_loss, 'val loss:', mean_val_loss))      
      
      if (mean_val_loss < best_val_loss) {
        best_val_loss <- mean_val_loss
        k <- 0  
      } else {
        k <- k + 1
        
        if (counter >= patience) {
          break
        }
      }
    }
  }
}


# getting model errors
as.data.frame(do.call(rbind, model_list)) %>% 
  cbind(params = names(model_list)) %>%
  cbind(max_epochs = sapply(model_list, function(x) length(x$`val_loss`))) %>% 
  group_by(params) %>%
  mutate(epoch = list(1:max_epochs)) %>% 
  unnest %>% 
  filter(epoch == max_epochs) %>% 
  arrange(val_loss) %>% 
  ungroup %>% 
  slice(1:3)


# making learning curves plot
as.data.frame(do.call(rbind, model_list)) %>% 
  cbind(params = names(model_list)) %>%
  cbind(max_epochs = sapply(model_list, function(x) length(x$`val_loss`))) %>% 
  group_by(params) %>%
  mutate(epoch = list(1:max_epochs)) %>% 
  unnest %>% 
  select(!max_epochs) %>% 
  pivot_longer(!c(epoch, params)) %>% 
  separate_wider_delim(params, delim = ' - ', names = c('lr', 'lambda')) %>% 
  ggplot(aes(x = epoch, y = value, color = name))+
  geom_line()+
  facet_wrap(~paste0('LR: ', lr, ' Lambda: ', lambda), nrow = 3) +
  labs(color = 'Data split', y = 'MSE loss')+
  ggtitle('Learning curves across hyperparameters')+
  theme_minimal()
# ggsave('2_plots/learning_curves.png')


## final model
# LR = 0.01, lambda = 0.00001

# defining model

set.seed(2025)
final_model <- ModelClass()

optimizer <- optim_sgd(final_model$parameters, lr = 0.01, weight_decay = 0.00001)
loss_func <- nnf_mse_loss

mean_train_losses <- c()
mean_val_losses <- c()


for (epoch in 1:20) {
  
  final_model$train()
  train_losses <- c()  
  
  coro::loop(for (b in train_dl) {
    optimizer$zero_grad()
    output <- final_model(b[[1]])
    loss <- loss_func(output, b[[2]])
    loss$backward()
    optimizer$step()
    train_losses <- c(train_losses, loss$item())
  })
  
  final_model$eval()
  valid_losses <- c()
  
  coro::loop(for (b in test_dl) {
    output <- final_model(b[[1]])
    loss <- loss_func(output, b[[2]])
    valid_losses <- c(valid_losses, loss$item())
  })
  
  mean_train_loss <- mean(train_losses)
  mean_val_loss <- mean(valid_losses)
  
  mean_train_losses <- c(mean_train_losses, mean_train_loss)
  mean_val_losses <- c(mean_val_losses, mean_val_loss)
  
  # print(paste('Epoch: ', epoch, 'train loss:', mean_train_loss, 'val loss:', mean_val_loss))      
  
}

# getting model preds on test and training data
with_no_grad(
  preds <- final_model(test_x)
)

with_no_grad(
  preds_training <- final_model(train_x)
)

model_predictions_on_test <- cbind(real = as.numeric(test_y), preds = as.numeric(preds)) %>% as.data.frame
model_preds_on_training <- cbind(real = as.numeric(train_y), preds = as.numeric(preds_training)) %>% as.data.frame



## alternative regression approach
# this is supervised learning with continuous output, so regression is our naive approach:

# training linear model
linear_model <- lm(target ~ ., data = df_train)

# getting preds on linear model for test and train
linear_preds <- predict(linear_model, df_test)
linear_preds_train <- predict(linear_model, df_train)


# ridge regression parameter grid
ridge_params <- c(0.01, 0.001, 0.0001, 0.00001)

# running ridge regression
ridge_regression <- glmnet(df_train %>% select(!target), df_train$target, alpha = 0, lambda = ridge_params)

# getting ridge regression predictions
preds_ridge <- predict(ridge_regression, s = 0.2, newx = df_test %>% select(!target) %>% as.matrix)

ridge_MSE <- list()
ridge_MAE <- list()

#looping through to get test errors
for (param in ridge_params){
  
  preds_ridge <- predict(ridge_regression, s = param, newx = df_test %>% select(!target) %>% as.matrix)
  
  MSE <- mean((as.numeric(preds_ridge) - as.numeric(df_test$target))**2)
  MAE <- mean(abs(preds_ridge - df_test$target))
  ridge_MSE[[paste(param)]] <- MSE
  ridge_MAE[[paste(param)]] <- MAE
}

# summarising models
do.call(rbind, ridge_MSE) %>% 
  cbind(names(ridge_MSE)) %>% 
  as.data.frame() %>% 
  rename(MSE = V1, lambda = V2) %>% 
  mutate_all(as.numeric) %>% 
  arrange(MSE)

do.call(rbind, ridge_MAE) %>% 
  cbind(names(ridge_MAE)) %>% 
  as.data.frame() %>% 
  rename(MSE = V1, lambda = V2) %>% 
  mutate_all(as.numeric) %>% 
  arrange(MSE)



## MODEL DIAGNOSTICS
# DNN on test
model_predictions_on_test %>%
  summarise(MSE = mean((real - preds)**2),
            MAE = mean(abs(real - preds)),
  )

# DNN on train
model_preds_on_training %>%
  summarise(MSE = mean((real - preds)**2),
            MAE = mean(abs(real - preds)),
  )

# OLS on test
cbind(real = df_test$target, 
      preds = linear_preds) %>% 
  as.data.frame %>% 
  summarise(MSE = mean((real - preds)**2),
            MAE = mean(abs(real - preds)),
  )

# OLS on train
cbind(real = df_train$target, 
      preds = linear_preds_train) %>% 
  as.data.frame %>% 
  summarise(MSE = mean((real - preds)**2),
            MAE = mean(abs(real - preds)),
  )


# making residuals plot
preds_comp_df <- model_predictions_on_test %>% mutate(model = 'DNN') %>% 
  rbind(cbind(real = df_test$target, preds = linear_preds, model = 'OLS'))

preds_comp_df %>% 
  mutate(real = as.numeric(real),
         preds = as.numeric(preds)) %>%
  ggplot(aes(x = preds, y = real-preds, color = model))+
  geom_point(size = 2, pch = 1)+
  scale_color_manual(values = c('steelblue', 'darkorange'))+
  labs(x = 'Predicted value', y = 'Residual', color = 'Model', title = 'Predicted values vs. residuals')+
  theme_minimal()
# ggsave('2_plots/preds_resids.png')



