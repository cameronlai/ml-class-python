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
#  linear exercise. You will need to complete short sections of code within the code below

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
