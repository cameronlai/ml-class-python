import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

## Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     warmUpExercise
#     plotData
#     gradientDescent
#     computeCost

# ==================== All function declaration ====================

def warmUpExercise():
    A = []
    # ============= YOUR CODE HERE =============
    # Instructions: Return the 5x5 identity matrix 
    # ===========================================
    return A

def plotData(x, y, label='Training Data'):
    # ============= YOUR CODE HERE =============
    # Instructions: Plot the training data into a figure
    # Hint: Set the correct label for the plot for legend display
    # ===========================================
    
def computeCost(X, y, theta):
    m = y.shape[0]
    J = 0
    # ============= YOUR CODE HERE =============
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    # ===========================================
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        # ============= YOUR CODE HERE =============
        # Instructions: Perform a single gradient step on the parameter vector theta. 
        # ===========================================

        # Save the cost J in every iteration 
        J_history[i] = computeCost(X, y, theta)
    return theta

if __name__ == "__main__":
    plt.close('all')
    plt.ion() # interactive mode

    # ==================== Part 1: Basic Function ====================

    print('Running warmUpExercise ...');
    print('5x5 Identity Matrix:');

    A = warmUpExercise()
    print(A)

    raw_input('Program paused. Press enter to continue')

    # ======================= Part 2: Plotting =======================

    print('Plotting Data ...')

    data_file = '../../data/ex1/ex1data1.txt'
    data = np.loadtxt(data_file, delimiter=',')

    x = data[:,0]
    y = data[:,1]
    m = data.shape[0] # number of training examples
    y = y.reshape((-1,1)) # create column matrix

    # Note: You have to complete the code in function plotData
    plotData(x,y, label='Training data')

    raw_input('Program paused. Press enter to continue')

    # =================== Part 3: Gradient descent ===================

    print('Running Gradient Descent ...')

    X = np.column_stack((np.ones(m), x))
    theta = np.zeros((2,1))

    iterations = 1500
    alpha = 0.01

    print(computeCost(X, y, theta))

    theta = gradientDescent(X, y, theta, alpha, iterations)

    print('Theta found by gradient descent: ');
    print(theta)

    plt.plot(X[:,1], np.dot(X,theta), '-', label='Linear Regression' )
    plt.legend()
    
    predict1 = np.dot([1, 3.5], theta)[0]
    print('For population = 35,000, we predict a profit of %f' % (predict1*10000));
    predict2 = np.dot([1, 7], theta)[0]
    print('For population = 70,000, we predict a profit of %f' % (predict2*10000));

    raw_input('Program paused. Press enter to continue')

    # ============= Part 4: Visualizing J(theta_0, theta_1) =============

    print('Visualizing J(theta_0, theta_1) ...')

    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    xv, yv = np.meshgrid(theta0_vals, theta1_vals)
    theta_vals = np.column_stack((xv.ravel(), yv.ravel()))

    J_vals = np.array([computeCost(X,y,t.reshape(-1,1)) for t in theta_vals])

    # Surface plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(theta_vals[:,0], theta_vals[:,1], J_vals, cmap=cm.jet, linewidth=0.2)
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')

    # Contour plot
    plt.figure()
    J_vals_contour = J_vals.reshape(-1, theta0_vals.shape[0])
    plt.contour(xv, yv, J_vals_contour, np.logspace(-2, 3, 20))
    plt.xlabel('theta_0')
    plt.ylabel('theta_1')
    plt.plot(theta[0,0], theta[1,0], 'rx', markersize=10, linewidth=2);
