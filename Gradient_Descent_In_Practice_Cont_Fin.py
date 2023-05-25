# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:00:12 2023

@author: emalin1
"""

########################## Feature Scaling #########################
#We scale because the gradient descent could jump from value to value due to very big or small numbers; it will run much slower
# 1 Way - Divide all numbers from a variable by the maximum 
# Mean Normalization - find the average of all the observations in a variable; 
    # Formula ((x1 - avg)/(max - min))
# Z-Score Normalization - calculate standard dviation and mean of each feature (variable);
    # Formula (x1 - avg)/st.dev.  
#As a rule of thumb we could aim from -1 to 1 for each feature Xj
    # Example: Feature going from 0 to 3 and another -2 to 0.5- that's ok no rescaling 
    # -100 to 100 : too large, needs rescaling 
    # -0.001 to 0.001: too small, needs rescaling 
    # When in doubt - better rescale so gradient descent could run much faster 

################## Check Gradient Descent is Really Finding Local Minimum (Converging) ######################### 
# Objective is to find optimal parameters that minimize the cost function 
# Learning Curve - x axis is the iterations and cost function is on the y; 
    # As we run more iterations the value of the cost function should be decreasing and converging  
    # the cost function should decrease after every iteration 
    # Iterations that are needed for convergence vary for each application 
# Automatic Convrgence test 
    # Let e (epsilon) is 0.001 
    # if the cost function decreases by epsilon in each iteration , declare convergence 

################## Chosing the Learning Rate ######################### 
# If Alpha is too big, it goes from small cost function to big or cost function keeps increasing 
# With small enough Alpha, cost function should decrease on every iteration 
# Better to start with a very small Alpha 
# However, if Alpha is too small thn gradient descent takes a lot more iterations to converge 
# Try range of values for Alphs (for example: 0.001, 0.01, 0.1, 1 ..... )
    # Then for each Alpha, graph Learning curve and see if the cost function is decreasing for each iteration 
    # Pick the largest possible learning rate that it doesn't take too much time but also cost function is decreasing for each iteration 

################## Chosing / Engineer the Most Important Features #########################
# Example - x1 * x2 as an area where x1 is lenght and x2 is wide 

################## Polynomial Regression #########################  
# F(x) = w1x1 + w2x2^2 + w3x3^3 + b 
    # Features x2 and x3 will be at very differnt scales from x1 so we need to use scaling if gradient descent 
# Square root function of a feature 
# What Features to use - next 

################## Selecting Features #########################  
 # We can check which features are more important by looking at the parameter Gradient Descent has chosen for us:
     # If the parameter is very small, close to 0 - not very important 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import  load_house_data, run_gradient_descent 
#from lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w
#from lab_utils_common import dlc

data=pd.read_csv("C:/Users/emalin1/houses.txt", sep=",", header = None)
data

x_train_pd = data.iloc[:, :4]
x_train = x_train_pd.to_numpy()

y_train_pd = data.iloc[:, 4]
y_train = y_train_pd.to_numpy()

x_features = ['size(sqft)','bedrooms','floors','age']

fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
len(ax)
for i in range(len(ax)):
    ax[i].scatter(x_train[:,i],y_train)
    ax[i].set_xlabel(x_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()

run_gradient_descent(x_train, y_train, 10, alpha = 9.9e-7)

mu = np.mean(x_train, axis=0)
sigma = np.std(x_train, axis = 0)

x_mean = x_train - mu
x_norm = (x_train - mu) / sigma

def zscore_normalize_features(x):
    mu = np.mean(x, axis = 0)
    sigma = np.std(x, axis = 0)
    x_norm = (x - mu) / sigma 
    
    return (x_norm, mu,sigma)

x_norm, mu, sigma = zscore_normalize_features(x_train)

############ Linear Regression using Scikit-Learn ##############
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

#Scale / normalize the training data 
x_norm = StandardScaler().fit_transform(x_train)

#Create the linear model - it will use Stochastic Gradient Descent 
sgdr = SGDRegressor(max_iter=1000)
#Fit the model
sgdr.fit(x_norm, y_train)

#View Parameters 
b_norm = sgdr.intercept_
w_norm = sgdr.coef_

#Make Predictions 
# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(x_norm)
# make a prediction using w,b. 
y_pred = np.dot(x_norm, w_norm) + b_norm 

# plot predictions and targets vs original features    
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(x_features[i])
    ax[i].scatter(x_train[:,i],y_pred,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

