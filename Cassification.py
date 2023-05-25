# -*- coding: utf-8 -*-
"""
Created on Sun May 14 10:08:34 2023

@author: EMALIN1
"""

########################### CLASSIFICATION ##############################
# You have categorical outcomes 
# Example: Binary Problems - outcome yes or no / 0 or 1 

########################### Logistic Regression ##############################
# It fits in sigmoid function / logistic function which is log like 
    # All positive on y axis
# Outputs b/w 0 and 1 
    # Log function: G(z) = 1 / (1 + e^-z) where 0 < G(z) < 1
    # if z is large, G(z) gets closer to 1 and when z is very negative, G(z) gets closer to 0 
    # if z = 0, then G(z) = 0.5 
# Logistic Regression Algorithm: 
    # F(x) = w * x + b = z
    # Pass that z to the logistic function: G(z) = 1 / (1 + e^-z)
    # F(x) = g(w*x + b) = 1 / (1 + e^(-(w*x + b)))
    # Outputs a number b/w 0 and 1 
# Interpretation of Logistic Regression Model: 
    # Probability that class is equal to 1 
    # if predicted F(x) = 0.7, then 70% chance that y is 1 
    # If y has a 70% chance of being 1, what is the chance that it is 0 ? 
        # y has to be either 0 or 1, thus the probability of bing 0 or 1 have to add up to 1 or 100% chance
        # P(y=0) + P(y=1) = 1, so if the chance of y to be 1 is 70% then the chance to be 0 is 30% 
        # F(x) = P(y=1|x;w,b) : probability that y is 1, given input x and parameters w,b
    
########################### Decision Boundaries ##############################
# Threshold of 0.5: if f(x) >= 0.5 then yhat = 1  
# Decision boundary when z = w*x + b = 0 because at that line it is almost neutral if yhat is 0 or 1
    # if  𝐰⋅𝐱+𝑏>=0, the model predicts  𝑦=1
    # if  𝐰⋅𝐱+𝑏<0, the model predicts  𝑦=0
# If for example z = w1x1 + w2x2 + b = 0 
    # w1x1 + w2x2 = -b so the decision boundary in this case is a line 
    # Anything to the ight of the line is 1 and anything to the lft is 0 
# Inside the shape would be 0 and outside 1 if for example the decision boundary is a circle  
 
########################## Cost Function - Logistic Regression ##############################   
# We need convex function in order to find the local minimum 
# Loss = L(F(x), y) where:
    # Loss if y = 1 then -log(F(x)) which is convex
    # Loss if y = 0 then -log(1 = F(x))
# Since log function can be only b/w 0 and 1, we only concentrate on that part of the function 
# when y = 1 (actual y), as F(x) approaches 1, loss is getting closer to 0 
# when y = 0 (actual y), as F(x) approaches 0, then loss is very small, close to 0 

########################## Simplified Cost Function ##############################   
# Since y could only be 1 or 0: 
    # L(F(x),y) = -y*log(F(x)) - (1-y)*log(1-F(x))
    # Cost Fn: J(w,b) = -1/m*sum overall all observations(y*log(F(x)) + (1-y)*log(1-F(x)))

########################## Gradient Descent Implementation ##############################   
#Used for finding optimal w and b 

# Vectorization and Feature scaling also apply to logistic regression

import numpy as np
import copy, math
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common import  dlc, plot_data, plt_tumor_data, sigmoid, compute_cost_logistic
from plt_quad_logistic import plt_quad_logistic, plt_prob
plt.style.use('./deeplearning.mplstyle')

#Example of Classification Problem 
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])

pos = y_train == 1
neg = y_train == 0
pos, neg

fig,ax = plt.subplots(1,2,figsize=(8,3))
ax[0]
#plot 1, single variable
ax[0].scatter(x_train[pos], y_train[pos], marker='x', s=80, c = 'red', label="y=1")
ax[0].scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", facecolors='none',lw=3)

ax[0].set_ylim(-0.08,1.1)
ax[0].set_ylabel('y', fontsize=12)
ax[0].set_xlabel('x', fontsize=12)
ax[0].set_title('one variable plot')
ax[0].legend()

#plot 2, two variables
plot_data(X_train2, y_train2, ax[1])
ax[1].axis([0, 4, 0, 4])
ax[1].set_ylabel('$x_1$', fontsize=12)
ax[1].set_xlabel('$x_0$', fontsize=12)
ax[1].set_title('two variable plot')
ax[1].legend()
plt.tight_layout()
plt.show()

import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common import  plot_data, sigmoid, dlc
plt.style.use('./deeplearning.mplstyle')

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])    
X_train
w = np.array([1,2])
w
X_train*w
np.dot(X_train,w)
X_train.shape
X_train[1]
np.dot(X_train[1],w)

#Compute the loss function of logarithmic regression 
def compute_cost_logistic(X,y,w,b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
       z_i = np.dot(X[i],w) + b
       f_wb_i = sigmoid(z_i)
       cost = cost + (-y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1 - f_wb_i))
    cost = cost / m
    return cost 

w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))

import matplotlib.pyplot as plt

# Choose values between 0 and 6
x0 = np.arange(0,6)

# Plot the two decision boundaries
x1 = 3 - x0
x1_other = 4 - x0

fig,ax = plt.subplots(1, 1, figsize=(4,4))
# Plot the decision boundary
ax.plot(x0,x1, label="$b$=-3")
ax.plot(x0,x1_other,  label="$b$=-4")
ax.axis([0, 4, 0, 4])

# Plot the original data
plot_data(X_train,y_train,ax)
ax.axis([0, 4, 0, 4])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.legend(loc="upper right")
plt.title("Decision Boundary")
plt.show()  

w_array1 = np.array([1,1])
b_1 = -3
w_array2 = np.array([1,1])
b_2 = -4

print("Cost for b = -3 : ", compute_cost_logistic(X_train, y_train, w_array1, b_1))
print("Cost for b = -4 : ", compute_cost_logistic(X_train, y_train, w_array2, b_2))

########################## Gradient Descent Implementation ##############################  
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

X_train
X_train.shape
X_train[3]
X_train[5, 1]

#Compute the loss function of logarithmic regression 
def compute_cost_logistic(X,y,w,b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
       z_i = np.dot(X[i],w) + b
       f_wb_i = 1/(1 + np.exp(-z_i))
       cost = cost + (-y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1 - f_wb_i))
    cost = cost / m
    return cost 


def compute_gradient_logistic(X, y, w, b): 
    m,n = X.shape
    dj_dw = np.zeros((n,)) 
    dj_db = 0
    
    for i in range(m): 
        z = np.dot(X[i],w) + b
        f_wb_i = 1/(1 + np.exp(-z))
        dj_db = dj_db + (f_wb_i - y[i])
        for j in range(n):
            dj_dw[j] = dj_dw[j] + (f_wb_i - y[i])*X[i,j]
                  
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw 

X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2.,3.])
b_tmp = 1.
dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X_tmp, y_tmp, w_tmp, b_tmp)
print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" 

def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    # An array to store cost J and w's at each iteration primarily for graphing later  
    J_history= []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)
        
        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic(X, y, w, b) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history         #return final w,b and J history for graphing

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")


########################## Scikit-Learn ##############################  
from sklearn.linear_model import LogisticRegression
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

lr_model = LogisticRegression()
lr_model.fit(X,y)

y_pred = lr_model.predict(X)
y_pred  #Prediction 

#Compute Accuracy 
lr_model.score(X,y)  
    











 