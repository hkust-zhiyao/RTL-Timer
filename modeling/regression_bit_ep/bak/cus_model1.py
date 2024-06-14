import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from lineartree import LinearBoostRegressor
from lineartree import LinearForestRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import pandas as pd
import os, time, json, copy, pickle
from multiprocessing import Pool
from random import shuffle
import numpy as np

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge


# def huber_approx_obj(dtrain, preds):
#     d = preds - dtrain.y
#     h = 1  #h is delta in the formula
#     scale = 1 + (d / h) ** 2
#     scale_sqrt = np.sqrt(scale)
#     grad = d / scale_sqrt
#     hess = 1 / scale / scale_sqrt
#     return grad, hess

def pseudo_huber_loss(y_pred, y_val):
    d = (y_val-y_pred)
    delta = 1  
    scale = 1 + (d / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt 
    hess = (1 / scale) / scale_sqrt
    return grad, hess

def custom_objective_old(y_pred, y_true):
    # Reshape y_pred to have three columns, one for each feature vector's prediction
    # predictions = y_pred.reshape(-1, 3)

    pred1 = y_pred[0:5]
    pred2 = y_pred[5:10]
    pred3 = y_pred[10:14]

    predictions = [pred1, pred2, pred3]
    
    # Take the max along axis=1 to get the final prediction
    y_pred_max = np.max(predictions, axis=1)
    
    # Compute the residuals
    residuals = y_pred_max - y_true
    
    # Gradient and hessian for squared error loss
    gradient = 2 * residuals
    hessian = 2 * np.ones_like(residuals)
    
    # Flatten gradient and hessian to match y_pred shape
    gradient = np.repeat(gradient, 3)
    hessian = np.repeat(hessian, 3)
    
    return gradient, hessian

def custom_objective(y_pred, y_true):
    # Reshape y_pred to have three columns, one for each feature vector's prediction
    predictions = y_pred.reshape(-1, 3)
    
    # Compute the average prediction along axis=1 for each sample
    y_pred_avg = np.mean(predictions, axis=1)
    
    # Compute the residuals
    residuals = y_pred_avg - y_true
    
    # Gradient and hessian for squared error loss
    gradient = 2 * residuals
    hessian = 2 * np.ones_like(residuals)
    
    # Flatten gradient and hessian to match y_pred shape
    gradient = np.repeat(gradient, 3)
    hessian = np.repeat(hessian, 3)
    
    return gradient, hessian

# Sample data: 100 samples, 3 feature vectors each of size 5
X = np.random.randn(100, 15)
y = np.random.randn(100)
# X = np.random.randn(100, 5).repeat(3, axis=0)  # Repeat each row 3 times
# y = np.random.randn(100).repeat(3)


parameters = {"objective": custom_objective_old,
              "n_estimators": 100,
              "eta": 0.3,
              "lambda": 1,
              "gamma": 0,
              "max_depth": None,
              "verbosity": 2}

    
model = xgb.XGBRegressor(**parameters)
model.fit(X, y)



# # Create estimators for stacking
# estimators = [
#     ('xgb', xgb.XGBRegressor()),
#     ('rf', RandomForestRegressor()),
#     ('rf2', RandomForestRegressor())
# ]

# # Create the stacking regressor
# stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=Ridge())

# # Fit the stacking regressor
# stacking_regressor.fit(X, y)

# # Make predictions on the test data
# y_pred = stacking_regressor.predict(X)

# Evaluate the model's performance
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)