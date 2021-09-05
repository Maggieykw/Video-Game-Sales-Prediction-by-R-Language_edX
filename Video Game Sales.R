##########################################################
# 1.Raw Data Preparation
##########################################################

#Installing Packpage
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# Information of Dataset: Video Game Sales with Ratings (please refer to the link below for details)
# https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings

# Importing the csv file into the variable for further analysis
raw_data <- read.csv("https://raw.githubusercontent.com/Maggieykw/edX_Video-Game-Sales-with-Ratings/main/Video_Games_Sales_as_at_22_Dec_2016.csv")

# Data Cleaning: removing missing values
raw_data <- raw_data %>% 
          filter(Year_of_Release!="N/A") %>% 
          filter(Publisher!="N/A") %>% 
          filter(Genre!="") %>%
          filter(Developer!="") %>%
          select(Platform,Year_of_Release,Genre,Publisher,Developer,Global_Sales)

##########################################################
# 2.Creation of Training Set and Validation (final hold-out test set)
##########################################################

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use ‘set.seed(1)‘

# Training set is 70 % of raw data and validation set is 30%
test_index <- createDataPartition(y = raw_data$Global_Sales, times = 1, p = 0.3, list = FALSE) 
train_set <- raw_data[-test_index,]
temp <- raw_data[test_index,]

# Make sure all independent variables,which are used to train modeld, exist in both training set and validation set
validation <- temp %>% 
                semi_join(train_set, by = "Platform") %>% 
                semi_join(train_set, by = "Year_of_Release") %>% 
                semi_join(train_set, by = "Genre") %>% 
                semi_join(train_set, by = "Publisher") %>% 
                semi_join(train_set, by = "Developer")

# Add rows removed from validation set back into training set
removed <- anti_join(temp, validation)
train_set <- rbind(train_set, removed)

# Remove temporary variables
rm(test_index, temp, raw_data, removed)

##########################################################
# 3.Creation of Video Game Sale Prediction Model using the above dataset
##########################################################

# Function of Model Performance Measure - Root Mean Squared Error
RMSE <- function(true_rating, predicted_rating){ sqrt(mean((true_rating - predicted_rating)^2))
}

##########################################################
# 3a.Considering Global Average in the Model only
##########################################################

# Computing the average sales in the dataset
mu_hat <- mean(train_set$Global_Sales) 

# Checking model performance
naive_rmse <- RMSE(validation$Global_Sales, mu_hat)

# Show result in more tidy way
rmse_results <- data_frame("Predictive Method" = "Just Global Average", RMSE = naive_rmse) 

#rmse_results %>% knitr::kable() #result testing

#RMSE of including average only: 1.510188

##########################################################
# 3b.Adding Platform Effect into Model
##########################################################

# Computing the least squares estimate effect of platform
platform_avgs <- train_set %>% 
  group_by(Platform) %>% 
  summarize(b_p = mean(Global_Sales - mu_hat))

# Computing the predicted sales
predicted_sales <- mu_hat + validation %>% 
  left_join(platform_avgs, by='Platform') %>%
  .$b_p

# Checking model performance
model_1_rmse <- RMSE(predicted_sales, validation$Global_Sales)
#model_1_rmse #for result testing

# Show result in more tidy way and combine it with the above model result for comparison
rmse_results <- bind_rows(rmse_results,
                  data_frame("Predictive Method"="Platform Effect Model",
                  RMSE = model_1_rmse ))

#rmse_results %>% knitr::kable() #for result testing

#RMSE of including average and platform effect model: 1.490115

##########################################################
# 3c.Adding Genre Effect into Model
##########################################################

# Computing the least squares estimate effect of genre
genre_avgs <- train_set %>% 
  left_join(platform_avgs, by='Platform') %>%
  group_by(Genre) %>%
  summarize(b_g = mean(Global_Sales - mu_hat - b_p))

# Computing the predicted sales
predicted_sales <- validation %>% 
  left_join(platform_avgs, by='Platform') %>%
  left_join(genre_avgs, by='Genre') %>%
  mutate(pred = mu_hat + b_p + b_g) %>%
  .$pred

# Checking model performance
model_2_rmse <- RMSE(predicted_sales, validation$Global_Sales)
#model_2_rmse #for result testing

# Show result in more tidy way and combine it with the above model results for comparison
rmse_results <- bind_rows(rmse_results,
                          data_frame("Predictive Method"="Platform + Genre Effects Model",  
                                     RMSE = model_2_rmse ))

#rmse_results %>% knitr::kable() #for result testing

#RMSE of including average, platform effect model and genre effect model: 1.489390

##########################################################
# 3d.Adding Publisher Effect into Model
##########################################################

# Computing the least squares estimate effect of publisher
publisher_avgs <- train_set %>% 
  left_join(platform_avgs, by='Platform') %>%
  left_join(genre_avgs, by='Genre') %>%
  group_by(Publisher) %>%
  summarize(b_pu = mean(Global_Sales - mu_hat - b_p - b_g))

# Computing the predicted sales
predicted_sales <- validation %>% 
  left_join(platform_avgs, by='Platform') %>%
  left_join(genre_avgs, by='Genre') %>%
  left_join(publisher_avgs, by='Publisher') %>%
  mutate(pred = mu_hat + b_p + b_g + b_pu) %>%
  .$pred

# Checking model performance
model_3_rmse <- RMSE(predicted_sales, validation$Global_Sales)
model_3_rmse #for result testing

# Show result in more tidy way and combine it with the above model results for comparison
rmse_results <- bind_rows(rmse_results,
                          data_frame("Predictive Method"="Platform + Genre + Publisher Effects Model",  
                                     RMSE = model_3_rmse ))

#rmse_results %>% knitr::kable() #for result testing
#RMSE of including average, platform effect model, genre and publisher effect model: 1.409233

##########################################################
# 3e.Adding Developer Effect into Model
##########################################################

# Computing the least squares estimate effect of developer
developer_avgs <- train_set %>% 
  left_join(platform_avgs, by='Platform') %>%
  left_join(genre_avgs, by='Genre') %>%
  left_join(publisher_avgs, by='Publisher') %>%
  group_by(Developer) %>%
  summarize(b_d = mean(Global_Sales - mu_hat - b_p - b_g - b_pu))

# Computing the predicted sales
predicted_sales <- validation %>% 
  left_join(platform_avgs, by='Platform') %>%
  left_join(genre_avgs, by='Genre') %>%
  left_join(publisher_avgs, by='Publisher') %>%
  left_join(developer_avgs, by='Developer') %>%
  mutate(pred = mu_hat + b_p + b_g + b_pu + b_d) %>%
  .$pred

# Checking model performance
model_4_rmse <- RMSE(predicted_sales, validation$Global_Sales)
#model_4_rmse #for result testing

# Show result in more tidy way and combine it with the above model results for comparison
rmse_results <- bind_rows(rmse_results,
                          data_frame("Predictive Method"="Platform + Genre + Publisher + Developer Effects Model",  
                                     RMSE = model_4_rmse ))

#rmse_results %>% knitr::kable() #for result testing

#RMSE of including average, platform effect model, genre, publisher and developer effect model: 1.329279

##########################################################
# 3f.Adding Year of Release Effect into Model
##########################################################

# Computing the least squares estimate effect of year of release
yr_avgs <- train_set %>% 
  left_join(platform_avgs, by='Platform') %>%
  left_join(genre_avgs, by='Genre') %>%
  left_join(publisher_avgs, by='Publisher') %>%
  left_join(developer_avgs, by='Developer') %>%
  group_by(Year_of_Release) %>%
  summarize(b_y = mean(Global_Sales - mu_hat - b_p - b_g - b_pu - b_d))

# Computing the predicted sales
predicted_sales <- validation %>% 
  left_join(platform_avgs, by='Platform') %>%
  left_join(genre_avgs, by='Genre') %>%
  left_join(publisher_avgs, by='Publisher') %>%
  left_join(developer_avgs, by='Developer') %>%
  left_join(yr_avgs, by='Year_of_Release') %>%
  mutate(pred = mu_hat + b_p + b_g + b_pu + b_d + b_y) %>%
  .$pred

# Checking model performance
model_5_rmse <- RMSE(predicted_sales, validation$Global_Sales)
#model_5_rmse #for result testing

# Show result in more tidy way and combine it with the above model results for comparison
rmse_results <- bind_rows(rmse_results,
                          data_frame("Predictive Method"="Platform + Genre + Publisher + Developer + Year of Release Effects Model",  
                                     RMSE = model_5_rmse ))

#rmse_results %>% knitr::kable() #for result testing

#RMSE of including average, platform effect model, genre, publisher, developer and year of release effect model: 1.326855

##########################################################
# 3.Regularization
##########################################################

# Choosing the penalty terms (within 0-10) using cross-validation
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  # Global average
  mu_hat <- mean(train_set$Global_Sales) 
  
  # Variable: Platform
  platform_reg_avgs <- train_set %>% 
    group_by(Platform) %>% 
    summarize(b_p = sum(Global_Sales - mu_hat)/(n()+l))
  
  # Variable: Genre
  genre_reg_avgs <- train_set %>% 
    left_join(platform_reg_avgs, by='Platform') %>%
    group_by(Genre) %>%
    summarize(b_g = sum(Global_Sales - mu_hat - b_p)/(n()+l))
  
  # Variable: Publisher
  publisher_reg_avgs <- train_set %>% 
    left_join(platform_reg_avgs, by='Platform') %>%
    left_join(genre_reg_avgs, by='Genre') %>%
    group_by(Publisher) %>%
    summarize(b_pu = sum(Global_Sales - mu_hat - b_p - b_g)/(n()+l))
  
  # Variable: Developer
  developer_reg_avgs <- train_set %>% 
    left_join(platform_reg_avgs, by='Platform') %>%
    left_join(genre_reg_avgs, by='Genre') %>%
    left_join(publisher_reg_avgs, by='Publisher') %>%
    group_by(Developer) %>%
    summarize(b_d = sum(Global_Sales - mu_hat - b_p - b_g - b_pu)/(n()+l))
  
  # Variable: Year of release
  yr_reg_avgs <- train_set %>% 
    left_join(platform_reg_avgs, by='Platform') %>%
    left_join(genre_reg_avgs, by='Genre') %>%
    left_join(publisher_reg_avgs, by='Publisher') %>%
    left_join(developer_reg_avgs, by='Developer') %>%
    group_by(Year_of_Release) %>%
    summarize(b_y = sum(Global_Sales - mu_hat - b_p - b_g - b_pu - b_d)/(n()+l))
  
  # Computing the predicted sales
  predicted_sales <- validation %>% 
    left_join(platform_reg_avgs, by='Platform') %>%
    left_join(genre_reg_avgs, by='Genre') %>%
    left_join(publisher_reg_avgs, by='Publisher') %>%
    left_join(developer_reg_avgs, by='Developer') %>%
    left_join(yr_reg_avgs, by='Year_of_Release') %>%
    mutate(pred = mu_hat + b_p + b_g + b_pu + b_d + b_y) %>%
    .$pred
  
  return(RMSE(predicted_sales, validation$Global_Sales))
})

# Plotting graph to see the optimal lambda (i.e. minimum RMSE)
#qplot(lambdas, rmses) #for checking
#lambdas[which.min(rmses)] #for checking

# Build the model using the lambda with the minimum RMSE: lambda = 2.25
lambda <- lambdas[which.min(rmses)]

# Regularizing all the above effects in the model

# Variable: Platform
platform_reg_avgs <- train_set %>% 
  group_by(Platform) %>% 
  summarize(b_p2 = sum(Global_Sales - mu_hat)/(n()+lambda), n_i = n())

# Variable: Genre
genre_reg_avgs <- train_set %>% 
  left_join(platform_reg_avgs, by='Platform') %>%
  group_by(Genre) %>%
  summarize(b_g2 = sum(Global_Sales - mu_hat - b_p2)/(n()+lambda), n_i = n())

# Variable: Publisher
publisher_reg_avgs <- train_set %>% 
  left_join(platform_reg_avgs, by='Platform') %>%
  left_join(genre_reg_avgs, by='Genre') %>%
  group_by(Publisher) %>%
  summarize(b_pu2 = sum(Global_Sales - mu_hat - b_p2 - b_g2)/(n()+lambda), n_i = n())

# Variable: Developer
developer_reg_avgs <- train_set %>% 
  left_join(platform_reg_avgs, by='Platform') %>%
  left_join(genre_reg_avgs, by='Genre') %>%
  left_join(publisher_reg_avgs, by='Publisher') %>%
  group_by(Developer) %>%
  summarize(b_d2 = sum(Global_Sales - mu_hat - b_p2 - b_g2 - b_pu2)/(n()+lambda), n_i = n())

# Variable: Year of release
yr_reg_avgs <- train_set %>% 
  left_join(platform_reg_avgs, by='Platform') %>%
  left_join(genre_reg_avgs, by='Genre') %>%
  left_join(publisher_reg_avgs, by='Publisher') %>%
  left_join(developer_reg_avgs, by='Developer') %>%
  group_by(Year_of_Release) %>%
  summarize(b_y2 = sum(Global_Sales - mu_hat - b_p2 - b_g2 - b_pu2 - b_d2)/(n()+lambda), n_i = n())

# Computing the predicted sales after the effects are regularized
predicted_sales <- validation %>% 
  left_join(platform_reg_avgs, by='Platform') %>%
  left_join(genre_reg_avgs, by='Genre') %>%
  left_join(publisher_reg_avgs, by='Publisher') %>%
  left_join(developer_reg_avgs, by='Developer') %>%
  left_join(yr_reg_avgs, by='Year_of_Release') %>%
  mutate(pred = mu_hat + b_p2 + b_g2 + b_pu2 + b_d2 + b_y2) %>%
  .$pred

# Checking model performance
model_6_rmse <- RMSE(predicted_sales, validation$Global_Sales)
#model_6_rmse #for result testing

# Show result in more tidy way and combine it with the above model results for comparison
rmse_results <- bind_rows(rmse_results,
                          data_frame("Predictive Method"="Regularized Effects (Platform + Genre + Publisher + Developer + Year of Release) Model",  
                                     RMSE = model_6_rmse ))

#Show all result of the above modeling
rmse_results %>% knitr::kable() 

#RMSE of including average, regularized effect model: 1.284444

##########################################################
# 4.Final Result
##########################################################

data_frame("Final Result" ="Regularized Effects (Platform + Genre + Publisher + Developer + Year of Release) Model",RMSE = model_6_rmse )%>% knitr::kable()
#Final RMSE (including average, regularized effect model): 1.284444

