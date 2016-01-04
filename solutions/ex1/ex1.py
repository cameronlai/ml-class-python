import numpy as np
import matplotlib.pyplot as plt

# ==================== All function declaration ====================

def warmUpExercise():
    A = []
    # ============= YOUR CODE HERE =============
    # Instructions: Return the 5x5 identity matrix 
    A = np.eye(5)
    # ===========================================
    return A

def plotData(x, y):
    # ============= YOUR CODE HERE =============
    # Instructions: Plot the training data into a figure
    plt.plot(x, y, 'rx', markersize=10)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    # ===========================================
    
def computeCost(X, y, theta):
    m = y.shape[0]
    J = 0
    # ============= YOUR CODE HERE =============
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    Jdiff = np.dot(X, theta) - y
    print(Jdiff)
    print(Jdiff.shape)
    J = np.sum(np.power(Jdiff, 2)) / (2*m)
    # ===========================================
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    J_history = np.zeros((num_iters, 1))
    # ============= YOUR CODE HERE =============
    # Instructions: Perform a single gradient step on the parameter vector theta. 
    for i in range(num_iters):
        tmpCost = np.sum(np.dot(X, theta), axis=0)
        theta[0] = theta[0] - alpha *  np.sum(tmpCost - y) / m 
        theta[1] = theta[1] - alpha *  np.sum((tmpCost - y) * x[1, :]) / m
        J_history[i] = computeCost(x, y, theta)
      # ===========================================
    return theta

if __name__ == "__main__":

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
    plotData(x,y)
    plt.show(block=False)

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

    plt.plot(x[1], np.sum(X*theta, axis=0), '-' )
    plt.show(block=False)

    raw_input('Program paused. Press enter to continue')

    # ============= Part 4: Visualizing J(theta_0, theta_1) =============

    print('Visualizing J(theta_0, theta_1) ...')
