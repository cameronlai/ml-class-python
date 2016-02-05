import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import optimize

## Machine Learning Online Class - Exercise 5: Regularized Linear Regression and Bias-Variance

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     linearRegCostFunction
#     learningCurve
#     validationCurve
#     polyFeatures
#

# ==================== All function declaration ====================

def linearRegCostFunction(X, y, theta, lambda_val):
    m = y.shape[0]
    J = 0
    # ============= YOUR CODE HERE =============
    # Instructions: Compute the cost and gradient of regularized linear 
    #               regression for a particular choice of theta.
    # ===========================================
    return J

def linearRegGradFunction(X, y, theta, lambda_val):
    m = y.shape[0]
    grad = np.zeros(theta.shape)
    # ============= YOUR CODE HERE =============
    # Instructions: Compute the cost and gradient of regularized linear 
    #               regression for a particular choice of theta.
    # ===========================================
    grad = grad.ravel()
    return grad

def trainLinearReg(X, y, lambda_val):
    initial_theta = np.zeros((X.shape[1], 1))

    costFunc = lambda t : linearRegCostFunction(X, y, t, lambda_val)
    gradFunc = lambda t : linearRegGradFunction(X, y, t, lambda_val)

    fmin_ret = optimize.fmin_cg(costFunc, initial_theta, gradFunc, maxiter=500, full_output=True)

    theta = fmin_ret[0]

    return theta
    
def learningCurve(X, y, Xval, yval, lambda_val):
    m = X.shape[0]
    error_train = np.zeros(m)
    error_val = np.zeros(m)
    # ============= YOUR CODE HERE =============
    # Instructions: Fill in this function to return training errors in 
    #               error_train and the cross validation errors in error_val. 
    # ===========================================
    return error_train, error_val
    
def polyFeatures(X, p):
    X_poly = np.zeros((X.size, p))
    # ============= YOUR CODE HERE =============
    # Instructions: Given a vector X, return a matrix X_poly where the p-th 
    #               column of X contains the values of X to the p-th
    #               power.
    # ===========================================
    return X_poly

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    sigma = np.std(X, axis=0)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma

def plotFit(min_x, max_x, mu, sigma, theta, p):
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1, 1)
    X_poly = polyFeatures(x, p)
    X_poly = (X_poly - mu) / sigma
    
    X_poly = np.column_stack((np.ones((x.shape[0],1)), X_poly))
    
    plt.plot(x, np.dot(X_poly, theta), 'b-')

def validationCurve(X, y, Xval, yval):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    error_train = np.zeros(lambda_vec.size)
    error_val = np.zeros(lambda_vec.size)
    # ============= YOUR CODE HERE =============
    # Instructions: Fill in this function to return training errors in 
    #               error_train and the validation errors in error_val. The 
    #               vector lambda_vec contains the different lambda parameters 
    #               to use for each calculation of the errors, i.e, 
    #               error_train(i), and error_val(i) should give 
    #               you the errors obtained after training with 
    #               lambda = lambda_vec(i)
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

    # =================== Part 2: Regularized Linear Regression Cost ===================

    theta = np.ones((2,1))
    X2 = np.column_stack((np.ones((m,1)), X))
    J = linearRegCostFunction(X2, y, theta, 1)
    
    print('Cost at theta = [1 ; 1]: %f \n (this value should be about 303.993192)' %  J)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 3: Regularized Linear Regression Gradient ===================

    theta = np.ones((2,1))
    grad = linearRegGradFunction(X2, y, theta, 1)
    
    print('Gradient at theta = [1 ; 1]: [%f; %f] \n (this value should be about [-15.303016; 598.250744])' %  (grad[0], grad[1]))

    raw_input('Program paused. Press enter to continue')

    # =================== Part 4: Train Linear Regression ===================

    lambda_val = 0;
    theta = trainLinearReg(X2, y, lambda_val);

    plt.plot(X, np.dot(X2, theta), 'b-', label='Linear Regression')
    plt.legend()

    raw_input('Program paused. Press enter to continue')

    # =================== Part 5: Learning Curve for Linear Regression ===================

    lambda_val = 0
    Xval2 = np.column_stack((np.ones((Xval.shape[0], 1)), Xval))
    error_train, error_val = learningCurve(X2, y, Xval2, yval, lambda_val)

    plt.figure()
    plt.title('Learning curve for linear regression')
    plt.legend('Train', 'Cross Validation')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.plot(np.arange(m), error_train, label='Train error')
    plt.plot(np.arange(m), error_val, label='Validation error')
    plt.legend()

    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in xrange(m):
        print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

    raw_input('Program paused. Press enter to continue')
    
    # =================== Part 6: Feature Mapping for Polynomial Regression ===================
    
    p = 8

    X_poly = polyFeatures(X, p)
    X_poly, mu, sigma = featureNormalize(X_poly)
    X_poly = np.column_stack((np.ones((m, 1)), X_poly))

    X_poly_test = polyFeatures(Xtest, p)
    X_poly_test = (X_poly_test - mu) / sigma
    X_poly_test = np.column_stack((np.ones((X_poly_test.shape[0], 1)), X_poly_test))

    X_poly_val = polyFeatures(Xval, p)
    X_poly_val = (X_poly_val - mu) / sigma
    X_poly_val = np.column_stack((np.ones((X_poly_val.shape[0], 1)), X_poly_val))

    print('Normalized Training Example 1:')
    print(X_poly[0, :])

    raw_input('Program paused. Press enter to continue')

    # =================== Part 7: Learning Curve for Polynomial Regression ===================

    lambda_val = 0
    theta = trainLinearReg(X_poly, y, lambda_val)
    
    plt.figure()
    plt.title('Polynomial Regression Fit (lambda = %f)' % lambda_val)
    plt.plot(X, y, 'rx', markersize=10, label='Training Data')
    plotFit(np.min(X), np.max(X), mu, sigma, theta, p)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')

    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_val)

    plt.figure()
    plt.title('Polynomial Regression Fit (lambda = %f)' % lambda_val)
    plt.legend('Train', 'Cross Validation')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.plot(np.arange(m), error_train, label='Train error')
    plt.plot(np.arange(m), error_val, label='Validation error')
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
    





