# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:12:02 2023

@author: emalin1
"""

########################## Scikit-Learn ##############################  
# If the model does not fit the data properly, it is underfitting or has high bias 
# Generalization - to pretty well but not perfectly; it can generalize to further observations / training examples (Just Right)
# High Variance or overfitting - fits the training set extremely well, the algoriths tries very hard to fit the training examples. 
    #So if there is an example that is slightly different , the function could be very different and give highly variable / different prediction

######################## Regularization to Reduce Overfitting ##############################  
# 1st option - If we can get more data, that is one way to reduce overfitting 
# 2nd option - include / exclude features to reduce overfitting; especially reduce features (Feature Selection)
    # Excluding features disadvantages - we can exclude some useful feature with useful data for the model
    # Need to be carful selecting the useful features (ways to do that)
# If we cannot do any of the above options, then try regularization 
# Regularization - reduce the size of parameters Wj 
    # Shrink the parameter rather than eliminating features 
    # Eliminating / reduce the effect of the features 
    # In practive there not much difference if you regularize B 
    # Penalize some features 
#Implementation: 
    # Because we don't know which parameters are the important ones, we penalize all of them 
    # First build a model with all featurs , and penalize all of them by shrinking all of them 
        # Add the term (lambda / 2m) * sum(Wj) over all j =1 to n number of featurs , add it to the cost - regularization term and lamba id the regularixation parameter
        # The value lambda gives a trade off of how you balance between the cost and the regularization term 
        # If lambda is 0 , then you get all the parameters and you can be overfitting 
        # If lambda is very , very large like 10^10 , then to minimize the term, you get all parametrs close to 0 so it will underfit 
        
    
######################## Linear Regression Regularization ############################## 
#Ssame Gradient Descent but now we add 1 more term 
    # Wj = Wj - alpha[(1/m) * sum over tot obs(F(w,b) - y)*Xj + ((lambda / m)*Wj)]
    # b = b - alpha * (1/m) * sum over all obs (F(w,b) - y)
    # Simultanoeus update for thee 2 
    
######################## Logistic Regression Regularization ##############################   
# Same as above, in the Gradient Descent and Cost Function we add the (lambda / 2*m) * sum of Wj^2 over all features X1 to Xj
# That reduces the optimal Wj
 
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from plt_overfit import overfit_example, output
from lab_utils_common import sigmoid
np.set_printoptions(precision=8)

def comput_cost_linear_reg(x, y, w, b, lambda_ = 1):
    err = 0 
    reg = 0 
    m, n = x.shape
    
    for i in range(m):
        fwb_i = np.dot(x[i],w) + b 
        err = err +(fwb_i - y[i])**2
    
    for j in range(n):
        reg += (w[j]**2)
        
    err = err / (2*m)
    reg = (lambda_ / (2*m))*reg
    total_cost = err + reg 
    return total_cost 
        
np.random.seed(1)       
x_tmp = np.random.rand(5,6)
x_tmp
y_tmp = np.array([0,1,0,1,0])
y_tmp
x_tmp[1]
w_tmp = np.random.rand(x_tmp.shape[1]).reshape(-1,)-0.5
w_tmp
b_tmp = 0.5
lambda_tmp = 0.7

cost_tmp = comput_cost_linear_reg(x_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
cost_tmp

def compute_cost_logistic_reg(x,y,w,b,lambda_ = 1): 
    
    m,n = x.shape
    cost = 0 
    for i in range(m):
        z_i = np.dot(x[i], w) + b 
        f_wb_i = 1 / ( 1 + np.exp(-z_i))
        cost += -y[i]*np.log(f_wb_i) - (1 - y[i])*np.log(1 - f_wb_i)
    cost = cost / m 
    
    reg_cost = 0 
    for j in range(n):
        reg_cost = reg_cost + (w[j]**2)
    reg_cost = (lambda_/(2*m)) * reg_cost 
    
    total_cost = cost + reg_cost 
    return total_cost
        
np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)

def compute_gradient_linear_reg(x,y,w,b,lambda_):
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0 
    reg = 0

    
    for i in range(m):
        err_1 = (np.dot(x[i],w) + b ) - y[i] 
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_1 * x[i,j]
        dj_db = dj_db + err_1
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_ / m)*w[j]
    return dj_db, dj_dw
        
np.random.seed(1)
x_tmp = np.random.rand(5,3)
x_tmp    
x_tmp.shape[0]        

y_tmp = np.array([0,1,0,1,0]) 
y_tmp
w_tmp = np.random.rand(x_tmp.shape[1])   
b_tmp = 0.5
lambda_tmp = 0.7

dj_db_tmp, dj_dw_tmp = compute_gradient_linear_reg(x_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
dj_db_tmp, dj_dw_tmp 

def compute_gradient_logistic_reg(x,y,w,b,lambda_):
    
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0 
    
    for i in range(m):
        z_i = np.dot(x[i], w) + b 
        f_wb_i = 1 / ( 1 + np.exp(-z_i))     
        err_i = f_wb_i - y[i]
        dj_db = dj_db + err_i
        for j in range(n): 
            dj_dw[j] = dj_dw[j] + err_i * x[i,j]
    dj_dw = dj_dw / m 
    dj_db = dj_db / m
    
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_ / m) * w[j]
        
    return dj_db, dj_dw 

np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )


