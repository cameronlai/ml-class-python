import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_ncg
from ex2 import *

## Machine Learning Online Class - Exercise 2: Logistic Regression

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#  mapFeature

# ==================== All function declaration ====================

def mapFeature(X1, X2):
    degree = 6
    retval = [np.ones(X1.shape)]
    for i in range(1, degree+1):
        for j in range(i+1):
            retval.append(np.power(X1, (i-j)) * np.power(X2, j))
    retval = np.transpose(np.array(retval))
    return retval

def costFunctionReg(theta, X, y, lambda_val):
    m = y.shape[0]
    J = 0
    grad = np.zeros(theta.shape)
    # ============= YOUR CODE HERE =============
    # Instructions:  Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta 
    J, grad = costFunction(theta, X, y)
    J = J + lambda_val * np.sum(np.power(theta[1:-1], 2)) / (2 * m)
    grad[1:] = grad[1:] + lambda_val * theta[1:] / m
    # ===========================================
    return J, grad

def plotDecisionBoundary(theta, X, y):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    uu, vv = np.meshgrid(u,v)
    uv_vals = np.column_stack((uu.ravel(), vv.ravel()))
    z = np.array([np.dot(mapFeature(uv_value[0], uv_value[1]), theta) for uv_value in uv_vals])
    z = z.reshape(-1, len(u))
    plt.contour(uu, vv, z, [0,0], colors='green', label='Decision Boundary')

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
    X = mapFeature(X[:,0], X[:,1])

    initial_theta = np.zeros(X.shape[1])

    # Set regularization parameter lambda to 1
    # As lambda is keyword in Python, it is replaced with lambda_val
    lambda_val = 1

    cost, grad = costFunctionReg(initial_theta, X, y, lambda_val)

    print('Cost at initial theta (zeros): %f' % cost);

    raw_input('Program paused. Press enter to continue')

    # =================== Part 2: Regularization and Accuracies ===================

    initial_theta = np.zeros((X.shape[1], 1))

    lambda_val = 1
    
    fmin_ret = fmin_ncg(lambda t : (costFunctionReg(t, X, y, lambda_val)[0]), initial_theta, lambda t : (costFunctionReg(t, X, y, lambda_val)[1]), maxiter=400, full_output=True)
    
    theta = fmin_ret[0]
    cost = fmin_ret[1]

    print('Cost at theta found by fmin: %f' % cost)
    print('theta:')
    print(theta)

    plotDecisionBoundary(theta, X, y)
    plt.title('lambda = %d' % lambda_val)
    plt.legend()

    p = predict(theta, X)

    print('Train Accuracy: %f' % (np.mean(p == y) * 100))

