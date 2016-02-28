import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from ex8_utility import *

## Machine Learning Online Class - Exercise 8: Anomaly Detection and Collaborative Filtering

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions 
#  in this exericse:
#
#     estimateGaussian
#     selectThreshold
#

# ==================== All function declaration ====================

def estimateGaussian(X):
    m, n = X.shape
    mu = np.zeros((n,1))
    sigma2 = np.zeros((n,1))
    # ============= YOUR CODE HERE =============
    # Instructions: Compute the mean of the data and the variances
    #               In particular, mu(i) should contain the mean of
    #               the data for the i-th feature and sigma2(i)
    #               should contain variance of the i-th feature.
    # ===========================================
    return mu, sigma2

def selectThreshold(yval, pval):
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    yval = yval.ravel()
    pval = pval.ravel()
    # ============= YOUR CODE HERE =============
    # Instructions: Compute the F1 score of choosing epsilon as the
    #               threshold and place the value in F1. The code at the
    #               end of the loop will compare the F1 score for this
    #               choice of epsilon and set it to be the best epsilon if
    #               it is better than the current choice of epsilon.
    # ===========================================
    return bestEpsilon, bestF1

if __name__ == "__main__":
    plt.close('all')
    plt.ion() # interactive mode

    # ==================== Part 1: Load Example Dataset ====================
    
    print('Visualizing example dataset for outlier detection.')

    data_file = '../../data/ex8/ex8data1.mat'
    mat_content = sio.loadmat(data_file)

    X = mat_content['X']
    Xval = mat_content['Xval']
    yval = mat_content['yval']

    plt.plot(X[:,0], X[:,1], 'bx')
    plt.xlim([0,30])
    plt.ylim([0,30])
    plt.xlabel('Latency (ms)');
    plt.ylabel('Throughput (mb/s)');

    raw_input('Program paused. Press enter to continue')

    # =================== Part 2: Estimate the dataset statistics ===================

    print('Visualizing Gaussian fit.')
    
    mu, sigma2 = estimateGaussian(X)

    p = multivariateGaussian(X, mu, sigma2)

    visualizeFit(X, mu, sigma2)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 3: Find Outliers ===================

    pval = multivariateGaussian(Xval, mu, sigma2)

    epsilon, F1 = selectThreshold(yval, pval)

    print('Best epsilon found using cross-validation: %e' % epsilon)
    print('Best F1 on Cross Validation Set:  %f' % F1)
    print('   (you should see a value epsilon of about 8.99e-05)')

    outliers = np.where(p < epsilon);

    plt.plot(X[outliers, 0], X[outliers, 1], 'ro')

    raw_input('Program paused. Press enter to continue')

    # =================== Part 4: Multidimensional Outliers ===================
    
    data_file = '../../data/ex8/ex8data2.mat'
    mat_content = sio.loadmat(data_file)

    X = mat_content['X']
    Xval = mat_content['Xval']
    yval = mat_content['yval']
    
    mu, sigma2 = estimateGaussian(X)
    
    # Training set
    p = multivariateGaussian(X, mu, sigma2)
    
    # Cross-validation set
    pval = multivariateGaussian(Xval, mu, sigma2)

    epsilon, F1 = selectThreshold(yval, pval)

    print('Best epsilon found using cross-validation: %e' % epsilon);
    print('Best F1 on Cross Validation Set:  %f' % F1);
    print('# Outliers found: %d' % np.sum(p < epsilon));
    print('   (you should see a value epsilon of about 1.38e-18)');    

    raw_input('Program paused. Press enter to continue')
    plt.close('all')
