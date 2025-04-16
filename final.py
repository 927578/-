#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:57:43 2022

@author: danieldu
"""
## Import everything needed.
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#=============================================================================#
#-------------------Part I: Reading Data & Rough Visualization----------------#
#=============================================================================#

# 1. Set the working path right where the data files exist.
print(os.getcwd())# show the current path.
os.chdir('/Users/danieldu/ECON380/FinalProject') #change the path.
print(os.getcwd()) # show the current path.

## 2. Read the train and test datasets.
train_df = pd.read_csv("train-1.csv")
test_df = pd.read_csv("test-2.csv")

## 3. Descriptive statistics
desc_stat = train_df.describe()

## 4. Display all distributions of numerical variables.
train_df.hist(bins=50, figsize=(20,10))
plt.show()

#=============================================================================#
#--------------------------Part II: Data Preparation--------------------------#
#=============================================================================#


## 1. Train set missing values handling.

train, test = train_test_split(train_df, test_size=0.2, random_state=20)
train.info()
# Count the missing number in train set.
count_missing_train = train.isnull().sum().sort_values(ascending=False)
# Transfer to percent type.
percent_missing_train = (train.isnull().sum() / train.isnull().count()
                         * 100).sort_values(ascending=False)
# Combine.
missing_data_train = pd.concat([count_missing_train, percent_missing_train],
                               axis=1)

# Drop variables that have 8% or above missing variables.
train.drop(percent_missing_train.index[percent_missing_train > 8].
           tolist(), axis=1, inplace=True)

# Check missing value again.
count_missing_train = train.isnull().sum().sort_values(ascending=False)
percent_missing_train = (train.isnull().sum() / train.isnull().count()
                         * 100).sort_values(ascending=False)
missing_data_train = pd.concat([count_missing_train, percent_missing_train],
                               axis=1)
# Get the still missing value.
still_missing = percent_missing_train.index[percent_missing_train>0].tolist()

#-----------------------------------------------------------------------------#


## 2. Fill missing values.


# Split the train set to numerical and categorical values.
train[still_missing].dtypes
train_num = train.select_dtypes(include=np.number)
train_cat = train.select_dtypes(exclude=np.number)

# Firstly, fill in the numerical values.

# Use median replacement to impute missing values for "height_in_cm".
imputer_median = SimpleImputer(strategy="median")
x = imputer_median.fit_transform(train_num[["height_in_cm"]])
train_num["height_in_cm"] = x

# Use the most frequent value to fill all missing values in train_cat.
train_cat.fillna(train_cat.mode().iloc[0], inplace=True)

# Combine train_num & train_cat back together as train.
train = pd.concat([train_num, train_cat], axis=1)

# Now, check if there is missing value again.
count_missing_train = train.isnull().sum().sort_values(ascending=False)
percent_missing_train = (train.isnull().sum() / train.isnull().count()
                         * 100).sort_values(ascending=False)
missing_data_train = pd.concat([count_missing_train, percent_missing_train],
                               axis=1)

# OK, there is no missing value in the train dataset.


#-----------------------------------------------------------------------------#



## 3. Select variables (feature engineering)


# (1) Age is useful for evaluating players' market value instead of the
#     date of birth. So, transfer the "date_of_birth" to "age".
train["date_of_birth"] = pd.to_datetime(train["date_of_birth"])
year_now = datetime.today().year
train["age"] = year_now - train.date_of_birth.dt.year
train = train.drop("date_of_birth", axis=1)


# (2) Split the train set to numerical and categorical values.
train_num = train.select_dtypes(include=np.number)
train_cat = train.select_dtypes(exclude=np.number)



# (3) Create two new variables that might be helpful for prediction.
#     "Goals per game" & "Assists per game"
train_num["goals_per_game"] = train_num["goals"] / train_num["total_appearance"]
train_num["assists_per_game"] = train_num["assists"] / train_num["total_appearance"]



# (4) Deal with skewed numeric features.

train_copy = train_num.copy()

# Check for the variable skewness.
train_skew = train_num.skew(axis=0)

# Get a list of highly skewed variables.
skewed_vars = train_skew.index[train_skew > 1].tolist()

# Split the train_num into two parts (low_sk & high_sk).
train_low_sk = train_copy.drop(skewed_vars, axis=1)
train_high_sk = train_num[skewed_vars]

# Use log-transform to adjust the highly skewed data.
adjust_skew = np.log(train_high_sk + 1)

# Merge.
train_num = pd.concat([train_low_sk, adjust_skew], axis=1)




# (5) Encoding categorical features with ordinal relationship.

from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
result = encoder.fit_transform(train_cat)

# According to OrdinalEncoder, Encoding rules are:
    # "foot": (Right, Left, Both) = (2, 1, 0)
    # "position": (Attack, Defender, Goalkeeper, Midfield) = (0, 1, 2, 3)
    
train_cat["foot"] = train_cat["foot"].replace(
    ["Right", "Left", "Both"], [2, 1, 0])

train_cat["position"] = train_cat["position"].replace(
    ["Attack", "Defender", "Goalkeeper", "Midfield"], [0, 1, 2, 3])

# Put them in a dataframe, and double check if all are numeric.
num_from_cat_vars = ["foot", "position"]
num_from_cat = train_cat[num_from_cat_vars]
num_from_cat.info()




# (6) Encoding categorical features without ordinal relationship.

# Drop the num_from_cat from train_cat.
# So train_cat will only have categorical variables.
train_cat = train_cat.drop(num_from_cat, axis=1)

# Use OneHotEncoder to convert them into dummies.
ohe = OneHotEncoder(handle_unknown='ignore')
dum_from_cat = pd.DataFrame(ohe.fit_transform(train_cat).toarray())

# Reforming the dataframe.
col_names = list(ohe.get_feature_names())
dum_from_cat.index = train_cat.index
dum_from_cat.columns = col_names

# Combine them to train_cat_to_num.
train_cat_to_num = pd.concat([num_from_cat, dum_from_cat], axis=1)
train_allnum = pd.concat([train_num, train_cat_to_num], axis=1)

# Double check if all variables are numeric.
still_cat = train_allnum.select_dtypes(exclude=np.number)
len(still_cat.columns)==0
# OK!


#-----------------------------------------------------------------------------#



## 4. Do everything above again to the test set.



test.info()
# Count the missing number in test set.
count_missing_test = test.isnull().sum().sort_values(ascending=False)
# Transfer to percent type.
percent_missing_test = (test.isnull().sum() / test.isnull().count()
                         * 100).sort_values(ascending=False)
# Combine.
missing_data_test = pd.concat([count_missing_test, percent_missing_test],
                               axis=1)

# Drop variables that have 8% or above missing variables.
test.drop(percent_missing_test.index[percent_missing_test > 8].
           tolist(), axis=1, inplace=True)

# Check missing value again.
count_missing_test = test.isnull().sum().sort_values(ascending=False)
percent_missing_test = (test.isnull().sum() / test.isnull().count()
                         * 100).sort_values(ascending=False)
missing_data_test = pd.concat([count_missing_test, percent_missing_test],
                               axis=1)
# Get the still missing value.
still_missing = percent_missing_test.index[percent_missing_test>0].tolist()

#-----------------------------------------------------------------------------#


## 2. Fill missing values.


# Split the train set to numerical and categorical values.
test[still_missing].dtypes
test_num = test.select_dtypes(include=np.number)
test_cat = test.select_dtypes(exclude=np.number)

# Firstly, fill in the numerical values.

# Use median replacement to impute missing values for "height_in_cm".
imputer_median = SimpleImputer(strategy="median")
x = imputer_median.fit_transform(test_num[["height_in_cm"]])
test_num["height_in_cm"] = x

# Use the most frequent value to fill all missing values in train_cat.
test_cat.fillna(test_cat.mode().iloc[0], inplace=True)

# Combine train_num & train_cat back together as train.
test = pd.concat([test_num, test_cat], axis=1)

# Now, check if there is missing value again.
count_missing_test = test.isnull().sum().sort_values(ascending=False)
percent_missing_test = (test.isnull().sum() / test.isnull().count()
                         * 100).sort_values(ascending=False)
missing_data_test = pd.concat([count_missing_test, percent_missing_test],
                               axis=1)

# OK, there is no missing value in the train dataset.


#-----------------------------------------------------------------------------#



## 3. Select variables (feature engineering)


# (1) Age is useful for evaluating players' market value instead of the
#     date of birth. So, transfer the "date_of_birth" to "age".
test["date_of_birth"] = pd.to_datetime(test["date_of_birth"])
year_now = datetime.today().year
test["age"] = year_now - test.date_of_birth.dt.year
test = test.drop("date_of_birth", axis=1)


# (2) Split the train set to numerical and categorical values.
test_num = test.select_dtypes(include=np.number)
test_cat = test.select_dtypes(exclude=np.number)



# (3) Create two new variables that might be helpful for prediction.
#     "Goals per game" & "Assists per game"
test_num["goals_per_game"] = test_num["goals"] / test_num["total_appearance"]
test_num["assists_per_game"] = test_num["assists"] / test_num["total_appearance"]



# (4) Deal with skewed numeric features.

test_copy = test_num.copy()

# Check for the variable skewness.
test_skew = test_num.skew(axis=0)

# Get a list of highly skewed variables.
skewed_vars_test = test_skew.index[test_skew > 1].tolist()

# Split the train_num into two parts (low_sk & high_sk).
test_low_sk = test_copy.drop(skewed_vars, axis=1)
test_high_sk = test_num[skewed_vars]

# Use log-transform to adjust the highly skewed data.
adjust_skew_test = np.log(test_high_sk + 1)

# Merge.
test_num = pd.concat([test_low_sk, adjust_skew_test], axis=1)




# (5) Encoding categorical features with ordinal relationship.

encoder = OrdinalEncoder()
result = encoder.fit_transform(test_cat)

# According to OrdinalEncoder, Encoding rules are:
    # "foot": (Right, Left, Both) = (2, 1, 0)
    # "position": (Attack, Defender, Goalkeeper, Midfield) = (0, 1, 2, 3)
    
test_cat["foot"] = test_cat["foot"].replace(
    ["Right", "Left", "Both"], [2, 1, 0])

test_cat["position"] = test_cat["position"].replace(
    ["Attack", "Defender", "Goalkeeper", "Midfield"], [0, 1, 2, 3])

# Put them in a dataframe, and double check if all are numeric.
num_from_cat_vars = ["foot", "position"]
num_from_cat = test_cat[num_from_cat_vars]
num_from_cat.info()




# (6) Encoding categorical features without ordinal relationship.

# Drop the num_from_cat from train_cat.
# So train_cat will only have categorical variables.
test_cat = test_cat.drop(num_from_cat, axis=1)

# Use OneHotEncoder to convert them into dummies.
ohe = OneHotEncoder(handle_unknown='ignore')
dum_from_cat = pd.DataFrame(ohe.fit_transform(test_cat).toarray())

# Reforming the dataframe.
col_names = list(ohe.get_feature_names())
dum_from_cat.index = test_cat.index
dum_from_cat.columns = col_names

# Combine them to test_cat_to_num.
test_cat_to_num = pd.concat([num_from_cat, dum_from_cat], axis=1)
test_allnum = pd.concat([test_num, test_cat_to_num], axis=1)

# Double check if all variables are numeric.
still_cat = test_allnum.select_dtypes(exclude=np.number)
len(still_cat.columns)==0
# OK!

## Now, the train and test datasets are all well prepared and become numeric.
## Check if they have the same number of features.
train_allnum.info()
test_allnum.info()



#=============================================================================#
#--------------------------Part III: Machine Learning-------------------------#
#=============================================================================#


## Scatter matrix among variables to the response "Market_value".


## 1. Set index, take the response variable we want to predict out and make 
##    train and test data matched.

train_allnum = train_allnum.set_index("player_id")
test_allnum = test_allnum.set_index("player_id")

# Drop the first column "Unnamed: 0".
train_allnum.drop(["Unnamed: 0"], axis=1, inplace=True)
test_allnum.drop(["Unnamed: 0"], axis=1, inplace=True)

# Also drop the columns which are helpless.
train_allnum.drop(["last_season", ""])

# Seperate data label from the features in the train and test.
train_label = train_allnum["market_value_usd"]
train_features = train_allnum.drop("market_value_usd", axis=1)

test_label = test_allnum["market_value_usd"]
test_features = test_allnum.drop("market_value_usd", axis=1)

# Sort the test features to be consistent with train features.
test_features = test_features[list(train_features.columns)]



## 2. Linear Regression Model

lin_reg = LinearRegression()
lin_reg.fit(train_features, train_label)

# Use train set to make prediction.
logMarkValue_train_linear = lin_reg.predict(train_features)
logMarkValue_train_linear = pd.Series(logMarkValue_train_linear)

# Report the MSE score and R^2 in linear regression of train data.
lin_mse_train = mean_squared_error(train_label, logMarkValue_train_linear)
lin_r2_train = r2_score(train_label, logMarkValue_train_linear)

# MSE = 0.8559, R^2 = 0.6386

# Convert the predicted logMarkValue back to Market Value (train).
predict_value_train_linear = np.expm1(logMarkValue_train_linear)

# Convert the observed logMarkValue back to Market Value (train).
observed_value_train = np.expm1(train_label)

# Use the function to see if the prediction is close (train).

def obs_vs_pred_train(observation, prediction):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(observation, prediction)
    ax.plot([0, max(observation)], [0, max(observation)], color='red')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    # ax.set_ylim(0, prediction.max())
    ax.set_xlabel("observed Sale Price")
    ax.set_ylabel("predicted Sale Price")
    ax.set_title("Train Set")
    plt.show()
    
obs_vs_pred_train(observed_value_train, predict_value_train_linear)

# Use the test set to make prediction.
logMarkValue_test_linear = lin_reg.predict(test_features)

# Report the MSE score and R^2 in linear regression of test data.
lin_mse_test = mean_squared_error(test_label, logMarkValue_test_linear)
lin_r2_test = r2_score(test_label, logMarkValue_test_linear)

# MSE = 0.8573, R^2 = 0.6204

# Convert the predicted logMarkValue back to Market Value (test).
predict_value_test_linear = np.expm1(logMarkValue_test_linear)

# Convert the observed logMarValue back to Market Value (test).
observed_value_test = np.expm1(test_label)

# See if the prediction is close (test).

def obs_vs_pred_test(observation, prediction):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(observation, prediction)
    ax.plot([0, max(observation)], [0, max(observation)], color='red')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    # ax.set_ylim(0, prediction.max())
    ax.set_xlabel("observed Sale Price")
    ax.set_ylabel("predicted Sale Price")
    ax.set_title("Test Set")
    plt.show()
    
obs_vs_pred_test(observed_value_test, predict_value_test_linear)

## The predicted value is not very close to the observed, and the model didn't 
## fit very well with not very low MSE and high R^2 close to 1. 
## .




## 3. LASSO. Try alpha = 0.5

lasso_reg = Lasso(alpha=0.5)
lasso_reg.fit(train_features, train_label)
logMarkValue_train_lasso = lasso_reg.predict(train_features)
logMarkValue_train_lasso = pd.Series(logMarkValue_train_lasso)

# Check the MSE and R^2 score.
lasso_mse_train = mean_squared_error(train_label, logMarkValue_train_lasso)
lasso_r2_train = r2_score(train_label, logMarkValue_train_lasso)
print("Lasso train model MSE score: ", lasso_mse_train)
print("Lasso train model R^2 score: ", lasso_r2_train)

# MSE = 1.4481, R^2 = 0.3886. 

predict_value_train_lasso = np.expm1(logMarkValue_train_lasso)
observed_price_train = np.expm1(train_label)
obs_vs_pred_train(observed_value_train, predict_value_train_lasso)

# Not good. See what happen if change alpha lower.




## 4. LASSO. Try alpha = 0.2

lasso_reg = Lasso(alpha=0.2)
lasso_reg.fit(train_features, train_label)
logMarkValue_train_lasso = lasso_reg.predict(train_features)
logMarkValue_train_lasso = pd.Series(logMarkValue_train_lasso)

# Check the MSE and R^2 score.
lasso_mse_train = mean_squared_error(train_label, logMarkValue_train_lasso)
lasso_r2_train = r2_score(train_label, logMarkValue_train_lasso)
print("Lasso train model MSE score: ", lasso_mse_train)
print("Lasso train model R^2 score: ", lasso_r2_train)

# MSE = 1.1948, R^2 = 0.4955, better but still not good. 

predict_value_train_lasso = np.expm1(logMarkValue_train_lasso)
observed_price_train = np.expm1(train_label)
obs_vs_pred_train(observed_value_train, predict_value_train_lasso)

# Not good.


### Lasso VS. Ridge in different levels of alpha.

result=pd.DataFrame(columns=["parameter","lasso train score","lasso test score",
                             "ridge train score","ridge test score"])
for i in range(1,100):
    alpha=i/10
    ridge=Ridge(alpha=alpha)
    lasso=Lasso(alpha=alpha,max_iter=10000)
    ridge.fit(train_features,train_label)
    lasso.fit(train_features,train_label)
    result=result.append([{"parameter":alpha,
                           "lasso train score":r2_score(train_label,lasso.predict(train_features)),
                           "lasso test score":r2_score(test_label,lasso.predict(test_features)),
                           "ridge train score":r2_score(train_label,ridge.predict(train_features)),
                           "ridge test score":r2_score(test_label,ridge.predict(test_features))}])

## Even though results show that Ridge performs better than Lasso in this case,
## the R^2 score of Ridge reaches only 0.62036 when alpha is 0.1.

## Check the predicted value VS. observed in Ridge regression.

# Train set.
ridge_reg = Ridge(alpha=0.1)
ridge_reg.fit(train_features, train_label)
logMarkValue_train_ridge = ridge_reg.predict(train_features)
logMarkValue_train_ridge = pd.Series(logMarkValue_train_ridge)

predict_value_train_ridge = np.expm1(logMarkValue_train_ridge)
obs_vs_pred_train(observed_value_train, predict_value_train_ridge)

# Test set.
ridge_reg.fit(test_features, test_label)
logMarkValue_test_ridge = ridge_reg.predict(test_features)
logMarkValue_test_ridge = pd.Series(logMarkValue_test_ridge)

predict_value_test_ridge = np.expm1(logMarkValue_test_ridge)
obs_vs_pred_test(observed_value_test, predict_value_test_ridge)

# The plots look very similar to linear regression model.

#### Important conclusion:
    ## If set alpha too small to let the model perform better, the effect of
    ## regularization would disappear and get the same result from OLS, which
    ## is from linear regression above.




## 5. Regression tree
tree_reg = DecisionTreeRegressor(random_state=42, max_depth=4)
tree_reg.fit(train_features, train_label)
logMarkValue_train_tree = tree_reg.predict(train_features)
logMarkValue_train_tree = pd.Series(logMarkValue_train_tree)
tree_mse_train = mean_squared_error(train_label, logMarkValue_train_tree)
tree_r2_train = r2_score(train_label, logMarkValue_train_tree)
print("Regression Tree model MSE score:", tree_mse_train)
print("Regression Tree model R^2 score:", tree_r2_train)


