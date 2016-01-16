import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn import linear_model

## Machine Learning Online Class
## Exercise 1: Linear Regression with multi variables by using scikit-learn

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     gradientDescentMulti
#     computeCostMulti
#     featureNormalize
#     normalEqn

# ==================== All function declaration ====================

def featureNormalize(X):
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    # ============= YOUR CODE HERE =============
    # Instructions: First, for each feature dimension, compute the mean
    #               of the feature and subtract it from the dataset,
    #               storing the mean value in mu. Next, compute the 
    #               standard deviation of each feature and divide
    #               each feature by it's standard deviation, storing
    #               the standard deviation in sigma. 
    # ===========================================
    return X_norm, mu, sigma
    
def computeCostMulti(X, y, theta):
    m = y.shape[0]
    J = 0
    # ============= YOUR CODE HERE =============
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    # ===========================================
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        # ============= YOUR CODE HERE =============
        # Instructions: Perform a single gradient step on the parameter vector theta. 
        # ===========================================

        # Save the cost J in every iteration 
        J_history[i] = computeCost(X, y, theta)
    return theta, J_history

def normalEqn(X, y):
    theta = np.zeros((X.shape[1], 1))
    # ============= YOUR CODE HERE =============
    # Instructions: Complete the code to compute the closed form solution to linear regression and put the result in theta.
    # ===========================================
    return theta

if __name__ == "__main__":
    plt.close('all')
    plt.ion() # interactive mode

    # ==================== Part 1: Creat Linear Model with Feature Normalization ====================

    print('Loading data ...');

    data_file = '../../data/ex1/ex1data2.txt'
    data = np.loadtxt(data_file, delimiter=',')

    X = data[:,0:2]
    y = data[:,2]
    m = data.shape[0] # number of training examples
    y = y.reshape((-1,1)) # create column matrix

    print('First 10 examples from the dataset:')
    print('X0%sX1%sy' % ('\t'*3,'\t'*3))
    print(data[0:11,:])
    
    raw_input('Program paused. Press enter to continue.');

    print('Normalizing Features ...')

    regr = None
    iterations = 400
    alpha = 0.01

    # ============= YOUR CODE HERE =============
    # Instructions: Use sklearn.LinearRegression.Ridge to creat a linear model with gradient descent and normalization
    # ===========================================
    
    if regr is None:
        sys.exit('Linear regression model not initialized')
    else:
        print(regr)
    
    raw_input('Program paused. Press enter to continue')

    # =================== Part 2: Run Linear Regression ===================

    print('Running Linear Regression ...')

    # ============= YOUR CODE HERE =============
    # Instructions: Use sklearn.LinearRegression to run linear regression
    # ===========================================    

    print('Coefficient found by linear_model: ');
    print(regr.coef_)
    print('Intercept found by linear_model: ');
    print(regr.intercept_)

    price = 0
    # ============= YOUR CODE HERE =============
    # Instructions: Estimate the price of a 1650 sq-ft, 3 br house
    # ===========================================
    
    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n %f' % price)

    raw_input('Program paused. Press enter to continue')
    
    # =================== Part 3: Normal Equations ===================

    print('Solving with normal equations...')

    regr = None

    # ============= YOUR CODE HERE =============
    # Instructions: Use sklearn.LinearRegression.Ridge to creat a linear model with singular value decomposition
    # ===========================================
    
    if regr is None:
        sys.exit('Linear regression model not initialized')
    else:
        print(regr)
    
    # Load data again
    data = np.loadtxt(data_file, delimiter=',')

    X = data[:,0:2]
    y = data[:,2]
    m = data.shape[0] # number of training examples
    y = y.reshape((-1,1)) # create column matrix

    # ============= YOUR CODE HERE =============
    # Instructions: Use sklearn.LinearRegression to run linear regression
    # ===========================================

    price = 0
    # ============= YOUR CODE HERE =============
    # Instructions: Estimate the price of a 1650 sq-ft, 3 br house
    # ===========================================

    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n %f' % price)
