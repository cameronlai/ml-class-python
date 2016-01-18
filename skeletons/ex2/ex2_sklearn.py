import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from ex2 import *

## Machine Learning Online Class - Exercise 2: Logistic Regression with sci-kit learn

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete a short section of code to perform logistic regression with scikit-learn library

if __name__ == "__main__":
    plt.close('all')
    plt.ion() # interactive mode

    # ==================== Part 1: Plotting ====================

    print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

    data_file = '../../data/ex2/ex2data1.txt'
    data = np.loadtxt(data_file, delimiter=',')

    X = data[:,0:2]
    y = data[:,2]
    m = data.shape[0] # number of training examples
    y = y.reshape((-1,1)) # create column matrix

    # Note: You have to complete the code in function plotData
    plotData(X,y, ['Admitted', 'Not admitted'])

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()

    raw_input('Program paused. Press enter to continue')

    # =================== Part 2: Logistic regression ===================

    # Note that C (inverse of regularization term) is specified to a larger value, rather than default 1.0
    # This means a weak regularization is used
    logistic = linear_model.LogisticRegression(C=1e5, max_iter=400)

    # ============= YOUR CODE HERE =============
    # Instructions: Use linear_model.LogisticRegression to run logistic regression
    # ===========================================

    print('Number of iterations used: %f' % logistic.n_iter_)

    print('Coefficient found by linear_model: ');
    print(logistic.coef_)
    print('Intercept found by linear_model: ');
    print(logistic.intercept_)

    plot_x = np.array([np.min(X[:,1]), np.max(X[:,1])])
    plot_y = (-1 / logistic.coef_[0, 1]) * (logistic.coef_[0, 0] * plot_x + logistic.intercept_)
    plt.plot(plot_x, plot_y, label='Decision Boundary')
    plt.legend()

    raw_input('Program paused. Press enter to continue')

    # ============= Part 3: Predict and Accuracies =============

    prob = logistic.predict_proba([[45, 85]])[0, 1]
    print('For a student with scores 45 and 85, we predict an admission probability of %f' % prob)

    print('Train Accuracy: %f' % logistic.score(X,y))
