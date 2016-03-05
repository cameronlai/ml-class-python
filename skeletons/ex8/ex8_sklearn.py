import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from ex8_utility import *
from sklearn.preprocessing import StandardScaler
from sklearn import grid_search
from sklearn import svm
from sklearn.metrics import f1_score
from scipy import stats

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

def trainClassifier(Xval, yval):
    yval = yval.ravel()
    
    outliers_fraction = yval.sum() / yval.size
    parameters = [{'nu': np.linspace(1, 7, 20) * outliers_fraction},
                    {'gamma': np.logspace(-3, 2, 50)}]

    clf = None
    # ============= YOUR CODE HERE =============
    # Instructions: Use sklearn one class SVM and GridSearchCV
    #               to find best threshold and best F1 score
    # ===========================================
    if clf is None:
        sys.exit('Model not initialized')

    print('Best estimator')
    print(clf.best_estimator_)

    yval_pred = clf.decision_function(Xval).ravel()  
    threshold = stats.scoreatpercentile(yval_pred, 100 * outliers_fraction)      
    yval_pred = yval_pred < threshold
    F1 = f1_score(yval, yval_pred)

    return clf, threshold, F1

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

    plot_datapoints(X)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 2: Estimate the dataset statistics ===================

    print('Visualizing Gaussian fit.')
    
    mu, sigma2 = estimateGaussian(X)

    p = multivariateGaussian(X, mu, sigma2)

    visualizeFit(X, mu, sigma2)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 3: Find Outliers with SVM ===================

    clf, threshold, F1 = trainClassifier(Xval, yval)

    print('Best threshold found using SVM: %e' % threshold)
    print('Best F1 on Cross Validation Set:  %f' % F1)

    y_pred = clf.decision_function(X).ravel()    
    y_pred = y_pred < threshold

    plt.figure()
    visualize_sklearn_clf(X, y_pred, threshold, clf)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 4: Multidimensional Outliers ===================
    
    data_file = '../../data/ex8/ex8data2.mat'
    mat_content = sio.loadmat(data_file)

    X = mat_content['X']
    Xval = mat_content['Xval']
    yval = mat_content['yval']
    
    clf, threshold, F1 = trainClassifier(Xval, yval)

    y_pred = clf.decision_function(X).ravel()    
    y_pred = y_pred < threshold

    print('Best threshold found using SVM: %e' % threshold)
    print('Best F1 on Cross Validation Set:  %f' % F1)
    print('# Outliers found: %d' % y_pred.sum());

    raw_input('Program paused. Press enter to continue')
    plt.close('all')
