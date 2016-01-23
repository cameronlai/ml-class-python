import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import linear_model
from ex3 import *

## Machine Learning Online Class - Exercise 3: Neural Network - One-vs-all with sci-kit learn

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  neural network exercise. You will need to complete a short section of code to perform logistic regression with scikit-learn library

if __name__ == "__main__":
    plt.close('all')
    plt.ion() # interactive mode

    # Setup the parameters you will use for this part of the exercies
    input_layer_size = 400 # 20x20 Input Images of Digits
    num_labels = 10 # 10 labels, from 1 to 10   
    # (note that we have mapped "0" to label 10)

    # ==================== Part 1: Loading and Visualizing Data ====================
    
    print('Loading and Visualizing Data ...')

    data_file = '../../data/ex3/ex3data1.mat'
    mat_content = sio.loadmat(data_file)

    X = mat_content['X']
    y = mat_content['y']

    m, n = X.shape

    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]

    displayData(sel)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 2: Vectorize Logistic Regression ===================

    print('Training One-vs-All Logistic Regression...')

    # Note that C is the inverse of regularization strength
    lambda_val = 0.1
    logistic = None

    # ============= YOUR CODE HERE =============
    # Instructions: Use LogisticRegression to find all_theta
    logistic = linear_model.LogisticRegression(C=1.0/lambda_val, max_iter=400)
    logistic.fit(X, y.ravel())
    # ===========================================

    if logistic is None:
        sys.exit('Logistic regression model not initialized')
    else:
        print(logistic)

    print('Number of iterations used: %f' % logistic.n_iter_)

    all_theta = np.column_stack((logistic.intercept_, logistic.coef_))

    raw_input('Program paused. Press enter to continue')

    # =================== Part 3: Vectorize Logistic Regression ===================

    pred = predictOneVsAll(all_theta, X)
    pred = pred.reshape(-1, 1)

    print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))
