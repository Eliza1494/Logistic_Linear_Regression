# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:23:00 2023

@author: EMALIN1
"""

################## Supervised Machine Learning #######################
# X to Y or Input to Output mappings 
#Give the leanring algorithms examples to learn from that have the right answers - output (Learns from being given right answers) 
#Example - Regression, Classification 
#Classification - multiple possible outcomesin terms of categories / classes ; they don't have to be numbers it could be non-numeric
        #The leanring algorith creates a boundary line between the 2 categories 

################## Unsupervised Machine Learning #######################
# Given data (inputs) but not mapped / assigned to particular output 
# So we try to find some structure or pattern in the data 
# Clustering Algorithms - put the data in different clusters 
# Anomaly Detection - used to detect / find unusual events (data points)
# Dimensionality Reduction - get bigger dataset and compress the data using fewer numbers 

################ Regression ########################
#Terminology: 
    # Training set - Data used to traing the model (input + output)
    # Input variable or feature - x 
    # Output variable or target - y 
    # m = number of training examples 
    # (x^i,y^i) = ith single training example 
    # Learning algorithm - function f (could be called hypothesis); takes x and gives yhat as prediction; model 
        # model prediction is called estimated y - what the true y could be 
    # When designing the model - what is the math formula for f 
    # Fw,b(x) = wx + b or F(x) or f(x) : linear regression with 1 variable (univariate linear regresssion)
    # Multivariate regression 
    
################ Cost Function ########################
# Fisrst step to linear regression is define cost function 
# Cost Function tells us how well the model is doing
# model f(x) = wx + b; w,b are called parameters or coefficients or weights 
# How well a line fits the training data - use cost function 
# yhat - y = Error ; then Error ^ 2 = square of the error 
# Measure the error across the entire training set / examples 
# Sum the Errors^2 across the training examples, then divide by m. If divide by 2 because for cleaner
# Other cost functions could be used but the above is the most popular  
# What does it do - we want to minimize the error (cost function) to find the optimal parameters w,b 
# Machine learning uses Gradient descent to determine the paramters that minimize the cost function (using derivatives)


import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

x_train.shape
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
m
print(f"Number of training examples is: {m}")

# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is: {m}")

i = 0 # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

#The scatter function arguments marker and c show the points as red crosses (the default is blue dots).
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()

w = 100
b = 100
print(f"w: {w}")
print(f"b: {b}")

#Calculate the output using the parameters w and b 
np.zeros(10)

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b)
tmp_f_wb

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

# Try to prdict 
w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")

import numpy as np
#%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
#plt.style.use('./deeplearning.mplstyle')

def compute_cost(x, y, w, b): 
    # number of training examples
    m = x.shape[0]    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])

plt_intuition(x_train,y_train)
plt.close('all') 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)