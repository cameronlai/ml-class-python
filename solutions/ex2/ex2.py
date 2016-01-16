import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

## Machine Learning Online Class - Exercise 2: Logistic Regression

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exericse:
#
#     plotData
#     sigmoid
#     costFunction
#     predict

# ==================== All function declaration ====================

def plotData(X, y, labels):
    # ============= YOUR CODE HERE =============
    # Instructions: Plot the positive and negative examples on a
    #               2D plot, using the option 'k+' for the positive
    #               examples and 'ko' for the negative examples.
    pos = np.where(y==1)[0]
    neg = np.where(y==0)[0]
    plt.plot(X[pos, 0], X[pos, 1], 'b+', label=labels[0])
    plt.plot(X[neg, 0], X[neg, 1], 'ro', label=labels[1])
    # ===========================================

def sigmoid(z):
    g = np.zeros(z.shape)
    # ============= YOUR CODE HERE =============
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
    g = 1 / (1 + np.exp(-z))
    # ===========================================
    return g

def costFunction(theta, X, y):
    m = y.shape[0]
    J = 0
    grad = np.zeros(theta.shape)
    # ============= YOUR CODE HERE =============
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    h_val = sigmoid(np.dot(X, theta)).reshape(-1,1)
    J = np.sum(-y * np.log(h_val) - (1-y) * np.log(1-h_val)) / m
    grad = np.sum((h_val-y) * X, axis=0) / m
    # ===========================================
    return J, grad
    
def predict(theta, X):
    m = X.shape[0]
    p = np.zeros((m, 1))
    # ============= YOUR CODE HERE =============
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters. 
    #               You should set p to a vector of 0's and 1's
    p[np.where(sigmoid(np.dot(X, theta)) > 0.5)] = 1
    # ===========================================
    return p

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

    # =================== Part 2: Compute Cost and Gradient ===================

    m, n = X.shape
    X = np.column_stack((np.ones(m), X))
    initial_theta = np.zeros((n+1,1))
    
    cost, grad = costFunction(initial_theta, X, y)
    
    print('Cost at initial theta (zeros): %f' % cost)
    print('Gradient at initial theta (zeros):')
    print(grad)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 3: Optimizing using fminunc ===================
    
    fmin_ret = fmin(lambda t : costFunction(t, X, y)[0], initial_theta, maxiter=400, full_output=True)
    
    theta = fmin_ret[0]
    cost = fmin_ret[1]

    print('Cost at theta found by fmin: %f' % cost)
    print('theta:')
    print(theta)

    plot_x = np.array([np.min(X[:,1]), np.max(X[:,1])])
    plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
    plt.plot(plot_x,plot_y, label='Decision Boundary')
    plt.legend()

    raw_input('Program paused. Press enter to continue')

    # ============= Part 4: Predict and Accuracies =============
    
    prob = sigmoid(np.dot([1, 45, 85], theta))
    print('For a student with scores 45 and 85, we predict an admission probability of %f' % prob)

    p = predict(theta, X)

    print('Train Accuracy: %f' % (np.mean(p == y) * 100))
