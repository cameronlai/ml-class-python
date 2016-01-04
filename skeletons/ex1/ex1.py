import numpy as np
import matplotlib.pyplot as plt

# ==================== All function declaration ====================

def warmUpExercise():
    A = []
    # ============= YOUR CODE HERE =============
    # Instructions: Return the 5x5 identity matrix 
    # ===========================================
    return A

def plotData(x, y):
    # ============= YOUR CODE HERE =============
    # Instructions: Plot the training data into a figure
    # ===========================================
    
def computeCost(x, y, theta):
    J = 0
    # ============= YOUR CODE HERE =============
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    # ===========================================
    return J

def gradientDescent(x, y, theta, alpha, num_iters):
    m = y.shape[0]
    J_history = np.zeros((num_iters, 1))
    # ============= YOUR CODE HERE =============
    # Instructions: Perform a single gradient step on the parameter vector theta. 
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
    m = data.shape[0]

    plotData(x,y)
    plt.show(block=False)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 3: Gradient descent ===================

    print('Running Gradient Descent ...')

    x = np.vstack((np.ones((1,m)), data[:,0]));
    theta = np.zeros((2,1))

    iterations = 1500
    alpha = 0.01

    print(computeCost(x, y, theta))

    theta = gradientDescent(x, y, theta, alpha, iterations)

    print('Theta found by gradient descent: ');
    print(theta)

    plt.plot(x[1], np.sum(x*theta, axis=0), '-' )
    plt.show(block=False)

    raw_input('Program paused. Press enter to continue')

    # ============= Part 4: Visualizing J(theta_0, theta_1) =============

    print('Visualizing J(theta_0, theta_1) ...')
