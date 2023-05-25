# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:07:23 2023

@author: emalin1
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

################ Gradient Descent ########################
# Alogorithm to resolve what parameters minimize the Cost Function 
# Used for linear regresssion and training neuro network models - deep learning models 
# Algorithm used to minimize any functions not just the Cost Function 
# Can work minimizing functions with more than 2 parameters 
# Finds the Local Minimas through the derivatives  
# Keep changing 1 parameter by keeping constat all other ones, and find the first derivative with parameter that gives 0 of the derivative 
# Gradient Descent Algorithms : w = w (old value of w) - alpha*the derivative wrt w ; 
    # Get the new value of w by adjusting the old value of w by some parameter * the derivative 
    # Alpha here is the leanring rate b/ 0 and 1 - it controls how much you move through the derivatives
    # In the calculation above for the derivative we use the old value of b 
# Gradient Descent Algorithms : b = b (old value of b) - alpha*the derivative wrt b ;
    # We use the non-updated values for b and w; the old value of w in the derivative 
# You repeat the algorithms (the above equations for each parameter) above wrt to parameters until the algorithm converges which will give us the min 
# We simultaneously update the parameters using the above algorithms until all parameter equations converge 

################ Gradient Descent - Derivatives ########################
# Partial derivative wrt each of the parameters 
# The derivative is the slope of a line with 1 variable
# It depends on alpha for where we start the derivative and convergence  

################ Gradient Descent - Alpha ########################
# Learning Rate 
# If the learning rate is very small - we get very small baby steps to updated value
    # You will need a lot of steps to get to the minimum 
    # Will neeed a very long time 
# What happens with a very large learning rate? 
    # Update the parameter with giant steps 
    # It could miss the minimum and actually get worse 
    # May fail to converge or diverge 
# If you have already reached the minimum - what happens ? 
    # If wrt w at the local minimum , the derivative term wtr w is equal to 0
    # If you have already reached minimum, the gradient descent just doesn't change the parameters 
# It can reach local minima with fixed learning rate 
# As we approach the derivative gets closer and closer to 0 (derivative converges to 0)

################ Gradient Descent - Linear Regression ########################  
# The Squared Error Cost Function for linear regression doesn't have multiple local minima
    # It has a single global minimum bc it is convex function (bowl shaped function)
# Batch Gradient Descent - in each step of the gradient descent we look at all the training examples instead of a subset 

import math, copy
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')
#from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
  
x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])

x_train.shape[0]

#Cost Function 
def compute_cost(x,y,w,b):
    
    m = x.shape[0]
    cost = 0 
    
    for i in range(m):
        f_wb = w*x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = (1 / (2*m)) * cost 
    
    return total_cost 

def compute_gradient(x, y, w, b): 

    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db

compute_gradient(x_train,y_train,0.5,1)

#Implement Graident Descent 
def gradient_descent(x,y,w_in,b_in,alpha,num_iters,cost_function, gradient_function):
    #Arrays to store cost J parameters at each iteration 
    J_history = []
    p_history = []
    b = b_in 
    w = w_in 
    
    for i in range(num_iters):
        #Calculate the gradient and update parameters using the gradient function (compute_gradient)
        dj_dw, dj_db = gradient_function(x,y,w,b)
        b = b - alpha*dj_db
        w = w - alpha*dj_dw
        
        #Save cost J at each iteration 
        if i <100000 :
            J_history.append(cost_function(x,y,w,b))
            p_history.append([w,b])
            
        # Print cost every at intervals 10 times or as many iterations if < 10    
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e}",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}", 
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w,b,J_history, p_history 

#Initialize Parameters         
w_init = 0 
b_init = 0 
#Gradient Descent settings 
iterations = 10000
tmp_alpha = 1.0e-2
#run gradient descent 
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train,
                                                    w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost,
                                                    compute_gradient)











 
