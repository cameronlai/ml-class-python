import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from ex2 import *
from ex2_reg import *

## Machine Learning Online Class - Exercise 2: Logistic Regression with sci-kit learn

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete a short section of code to perform logistic regression with scikit-learn library

if __name__ == "__main__":
    plt.close('all')
    plt.ion() # interactive mode

    data_file = '../../data/ex2/ex2data2.txt'
    data = np.loadtxt(data_file, delimiter=',')

    X = data[:,0:2]
    y = data[:,2]
    m = data.shape[0] # number of training examples
    y = y.reshape((-1,1)) # create column matrix

    # Note: You have to complete the code in function plotData
    plotData(X, y, ['y = 1', 'y = 0'])

    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()

    # ==================== Part 1: Regularized Logistic Regression ====================

    # Add Polynomial Features
    polynomial_features = PolynomialFeatures(degree=6, include_bias=False)
    X = polynomial_features.fit_transform(X)

    # Note that C is the inverse of regularization strength
    lambda_val = 1
    logistic = linear_model.LogisticRegression(C=1.0/lambda_val, max_iter=400)
    logistic.fit(X, y.ravel())

    print('Number of iterations used: %f' % logistic.n_iter_)
    print('Coefficient found by linear_model: ');
    print(logistic.coef_)
    print('Intercept found by linear_model: ');
    print(logistic.intercept_)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 2: Regularization and Accuracies ===================

    theta = np.hstack((logistic.intercept_[0], logistic.coef_[0]))

    plotDecisionBoundary(theta, X, y)
    plt.title('lambda = %d' % lambda_val)
    plt.legend()

    print('Train Accuracy: %f' % logistic.score(X,y))

