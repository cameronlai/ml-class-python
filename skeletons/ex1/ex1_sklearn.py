import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

## Machine Learning Online Class - Exercise 1: Linear Regression by using scikit-learn

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete a short section of code to perform linear regression with scikit-learn library

if __name__ == "__main__":
    plt.close('all')
    plt.ion() # interactive mode

    # ======================= Part 1: Plotting =======================

    print('Plotting Data ...')

    data_file = '../../data/ex1/ex1data1.txt'
    data = np.loadtxt(data_file, delimiter=',')

    x = data[:,0]
    y = data[:,1]
    m = data.shape[0] # number of training examples
    x = x.reshape(m, -1)
    y = y.reshape(m, -1)

    plt.plot(x, y, 'rx', markersize=10, label='Training data')
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')

    raw_input('Program paused. Press enter to continue')

    # =================== Part 2: Run Linear Regression ===================

    print('Running Linear Regression ...')

    iterations = 400
    alpha = 0.01

    # Ridge regression in sci-kit learn includes all linear regression with regularization
    # Using 'auto' can allow selection of solver automatically
    regr = linear_model.Ridge(alpha=alpha, max_iter=iterations, solver='sag')

    # ============= YOUR CODE HERE =============
    # Instructions: Use sklearn.LinearRegression to run linear regression
    # ===========================================

    print('Coefficient found by linear_model: ');
    print(regr.coef_)
    print('Intercept found by linear_model: ');
    print(regr.intercept_)

    plt.plot(x, regr.predict(x), '-', label='Linear Regression' )
    plt.legend()
    
    predict1 = regr.predict(3.5)
    print('For population = 35,000, we predict a profit of %f' % (predict1*10000));
    predict2 = regr.predict(7)
    print('For population = 70,000, we predict a profit of %f' % (predict2*10000));
