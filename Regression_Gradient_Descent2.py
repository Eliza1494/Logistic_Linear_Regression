# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 19:29:55 2023

@author: EMALIN1
"""

########################### Regression Model with multiple features ############################
    # Fw,b(X) = 0.1x1 + 4x2 + 10x3 +(-2)x4 + 80
    # For each additional x1 the y increases by 0.1
    # if no x1, x2, x3, x4 then y = 80 
    # row of vector w = [0.1 4 10 -2]
    # vector x = [x1 x2 x3 x4]
    # Then F vector w,b(vector x) = the dot product of w and x (w . x) + b
    # Vectorization above 
    # Miltivariate or Multiple features regression model 
    
######################## Vectorization ###############################
    # Vectorized code is quicker and more efficient 
    # row of vector w = [w1 w2 w3] and vector x = [x1 x2 x3]

######################## Normal Equation ###############################
# Alternative to Gradient Descent
# Only used for Linear Regression
# Doesn't generalize to other larning algorithsm
# Slow when number of features is large
# It may be used in machine learning libraries that implement linear regression to solve w and b
# Gradient descent is usually better


import numpy as np    # it is an unofficial standard to use np for numpy
import time
#Vectors:
w = np.array([1.0, 2.5, -3.3])
b = 4 
x = np.array([10,20,30])
w[0] 
x[0]

#Compute the dot product b/w the vectors w and x
f = np.dot(w, x) + b    
f

#Without vectorization - way slower and less efficient: 
f1 = 0
for j in range(0,3):
    f1 = f1 + w[j] * x[j]
f1   

a = np.zeros(4)
a
a = np.zeros(4);                print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
b = np.zeros((4,));             print(f"np.zeros(4,) :  b = {a}, b shape = {b.shape}, b data type = {b.dtype}")
c = np.random.random_sample(4); print(f"np.random.random_sample(4): c = {c}, c shape = {c.shape}, c data type = {c.dtype}")

# NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
a = np.arange(4.);              print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4);          print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

 # NumPy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

#vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)

a[0]
a[2]
a[-1]

#access 5 consecutive elements (start:stop:step)
a[2:6:2]
a[6]
a[2:7:2]

# access all elements index 3 and above
a[3:]

# access all elements below index 3
a[:3]

#Acess all elements 
a[:]

a = np.array([1,2,3,4])

#Begate the elements of a
b = -a
b

#sum all elements
np.sum(a)
np.mean(a)
a*2
np.sum(a[1:4])

a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
a+b

#Vectors need to be the same size
#c = np.array([1, 2])
#a+c

a = np.array([1, 2, 3, 4])
# multiply a by a scalar
b = 5 * a 

#Dot product for loop (vectors need to be the same size)

def my_dot(a,b):
    x = 0 
    for i in range(a.shape[0]):
        x = x + (a[i] * b[i])
    return x

a1 = np.array([1, 2, 3, 4])
b1 = np.array([-1, 4, 3, 2])

my_dot(a1,b1)

#Faster 
c1 = np.dot(a1, b1) 
c1   

X = np.array([[1],[2],[3],[4]])
w = np.array([2])
c = np.dot(X[1], w)

#Creating Matrices
am = np.zeros((1, 5))  
am
am2 = np.zeros((2, 5))  
am2
print(f"am2_shape = {am2.shape}, am2 = {am2}") 
am2 = np.random.random_sample((1, 1))  
am2

am3 = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
am3[2,0]

am4 = np.arange(24).reshape(4, 6)   #reshape is a convenient way to create matrices
am4
am4[2,4]
am4[3]

#vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

a[0, 2:7:1]
a[:, 2:7:1]
a[:,:]
a[1,:]
#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")



####################### Multiple Linear Regression Model #########################
import copy, math
import numpy as np
import matplotlib.pyplot as plt

#[2104, 5, 1, 45] is 1 row; so 2104 is house size, 5 is number of bedrooms, 1 is number of floors, 45 age of home
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

X_train.shape[0]
X_train[2]
#Single Prediction - element by element 
def predict_single_loop(x,w,b):
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i]*w[i]
        p = p_i + p
    p = p + b
    return p 

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
w_init[2]
X_train[2] * w_init[2]
X_train[2] * w_init

#single prediction 
x_vec = X_train[0,:]
x_vec
predict_single_loop(x_vec, w_init, b_init) #assuming w_init and b_init are the optimal

#Easier 
p = np.dot(x_vec, w_init) + b_init
p

# Use the Gradient Descent Formula to find optimals 
def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0 
    for i in range(m):
        yhat_i = np.dot(x[i],w) + b
        cost = cost + (yhat_i - y[i])**2  
    cost = cost / 2*m
    return cost

compute_cost(X_train, y_train, w_init, b_init)

X_train.shape

####################### Gradient Descent with Multiple Variables #########################
X_train[1]
X_train.shape
def compute_gradient(x,y,w,b):
    m,n = x.shape
    dj_dw = np.zeros((n,)) #Set the derivative to 0 
    dj_db = 0 
    
    for i in range(m):
        err = (np.dot(x[i],w)+b) - y[i]
        for j in range (n):
            dj_dw[j] =  dj_dw[j] + (err * x[i, j])
        dj_db = dj_db + err 
    dj_dw = dj_dw / m #Computes the derivative for each variable 
    dj_db = dj_db / m 
    
    return dj_db, dj_dw

compute_gradient(X_train, y_train, w_init, b_init)


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 

    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing
                                    
 # initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations) 

w_final
b_final
J_hist          
