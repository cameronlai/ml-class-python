import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve, validation_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import make_scorer, mean_squared_error

## Machine Learning Online Class - Exercise 5: Regularized Linear Regression and Bias-Variance

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete a short section of code to perform linear regression with scikit-learn library
#

# ==================== All function declaration ====================

def polyFeatures(X, p):
    X_poly = np.zeros((X.size, p))
    # ============= YOUR CODE HERE =============
    # Instructions: Given a vector X, use sci-kit learn to generate polynomial features
    polynomial_features = PolynomialFeatures(degree=p, include_bias=False) # bias will be added in regression model
    # ===========================================
    return X_poly

def learningCurve(X, y, Xval, yval, lambda_val):
    train_size = X.shape[0]
    error_train = np.zeros(train_size)
    error_val = np.zeros(train_size)

    Xin = np.vstack((X, Xval))
    yin = np.vstack((y, yval))
    m, n = Xin.shape    
    # ============= YOUR CODE HERE =============
    # Instructions: Use sklearn.learning_curve to get learning_curve
                    np.arange(train_size, m)]]) # Fixed generator
    # ===========================================
    return error_train, error_val

def validationCurve(X, y, Xval, yval):
    lambda_vec = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    train_size = X.shape[0]
    error_train = np.zeros(train_size)
    error_val = np.zeros(train_size)

    Xin = np.vstack((X, Xval))
    yin = np.vstack((y, yval))
    m, n = Xin.shape    
    # ============= YOUR CODE HERE =============
    # Instructions: Use sklearn.learning_curve to get validation_curve
                    np.arange(train_size, m)]]) # Fixed generator
    # ===========================================
    return lambda_vec, error_train, error_val

if __name__ == "__main__":
    plt.close('all')
    plt.ion() # interactive mode

    # ==================== Part 1: Loading and Visualizing Data ====================
    
    print('Loading and Visualizing Data ...')

    data_file = '../../data/ex5/ex5data1.mat'
    mat_content = sio.loadmat(data_file)

    X = mat_content['X']
    y = mat_content['y']
    Xval = mat_content['Xval']
    yval = mat_content['yval']
    Xtest = mat_content['Xtest']
    ytest = mat_content['ytest']

    m, n = X.shape

    plt.plot(X, y, 'rx', markersize=10, label='Training Data')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')

    raw_input('Program paused. Press enter to continue')

    # =================== Part 2: Train Linear Regression ===================

    lambda_val = 0
    regr = None
    # ============= YOUR CODE HERE =============
    # Instructions: Use sci-kit learn to perform linear regression training
    # ===========================================
    if regr is None:
        sys.exit('Linear regression model not initialized')
    else:
        print(regr)
        
    plt.plot(X, regr.predict(X), 'b-', label='Linear Regression')
    plt.legend()

    raw_input('Program paused. Press enter to continue')
    
    # =================== Part 3: Learning Curve for Linear Regression ===================

    error_train, error_val = learningCurve(X, y, Xval, yval, lambda_val)

    plt.figure()
    plt.title('Learning curve for linear regression')
    plt.legend('Train', 'Cross Validation')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.plot(np.arange(1, m+1), error_train, label='Train error')
    plt.plot(np.arange(1, m+1), error_val, label='Validation error')
    plt.legend()

    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in xrange(m):
        print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

    raw_input('Program paused. Press enter to continue')
    

    # =================== Part 4: Feature Mapping for Polynomial Regression ===================
    
    p = 8

    X_poly = polyFeatures(X, p)
    X_poly_val = polyFeatures(Xval, p)
    X_poly_test = polyFeatures(Xtest, p)

    X_all = np.vstack((X, Xval, Xtest))
    X_all_poly = np.vstack((X_poly, X_poly_val, X_poly_test))
    y_all = np.vstack((y, yval, ytest))

    print('Normalized Training Example 1:')
    print(X_poly[0, :])

    raw_input('Program paused. Press enter to continue')

    # =================== Part 5: Learning Curve for Polynomial Regression ===================

    regr.fit(X_poly, y)
    
    plt.figure()
    plt.title('Polynomial Regression Fit (lambda = %f)' % lambda_val)
    plt.plot(X, y, 'rx', markersize=10, label='Training Data')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')

    X_plot = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)
    X_poly_plot = polyFeatures(X_plot, p)
    plt.plot(X_plot, regr.predict(X_poly_plot), 'b-', label='Linear Regression')

    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_val)

    plt.figure()
    plt.title('Polynomial Regression Fit (lambda = %f)' % lambda_val)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.plot(np.arange(1, m+1), error_train, label='Train error')
    plt.plot(np.arange(1, m+1), error_val, label='Validation error')
    plt.legend()

    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in xrange(m):
        print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

    raw_input('Program paused. Press enter to continue')

    # =================== Part 8: Validation for Selecting Lambda ===================

    lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

    plt.close('all')
    plt.figure()
    plt.xlabel('lambda')
    plt.ylabel('Error')
    plt.plot(lambda_vec, error_train, label='Train')
    plt.plot(lambda_vec, error_val, label='Cross Validation')
    plt.legend()

    print('# lambda\tTrain Error\tCross Validation Error')
    for i in xrange(lambda_vec.shape[0]):
        print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

    raw_input('Program paused. Press enter to continue')
    
    plt.close('all')
    





