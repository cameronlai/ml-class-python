import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import optimize
from ex4_utility import *

## Machine Learning Online Class - Exercise 4: Neural Network Learning

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoidGradient.m
#     randInitializeWeights.m
#     nnCostFunction.m
#

# ==================== All function declaration ====================

def reshape_param(nn_params, input_layer_size, hidden_layer_size, num_labels):
    # Reshape nn_params back into parameters Theta1 and Theta2
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)]
    Theta1 = Theta1.reshape((hidden_layer_size, input_layer_size + 1))

    Theta2 = nn_params[(hidden_layer_size * (input_layer_size + 1)):]
    Theta2 = Theta2.reshape((num_labels, hidden_layer_size + 1))
    return Theta1, Theta2

def predict(Theta1, Theta2, X):
    X = np.column_stack((np.ones(m), X))
    z2_val = np.dot(X, np.transpose(Theta1))
    hiddenLayer = sigmoid(z2_val)
    hiddenLayer = np.column_stack((np.ones(m), hiddenLayer))
    outputLayer = sigmoid(np.dot(hiddenLayer, np.transpose(Theta2)))
    p = np.argmax(outputLayer, axis=1) + 1
    p = p.reshape(-1, 1)
    return p

def sigmoid(z):
    g = np.zeros(z.shape)
    g = 1 / (1 + np.exp(-z))
    return g

def sigmoidGradient(z):
    g = np.zeros(z.shape)
    # ============= YOUR CODE HERE =============
    # Instructions: Compute the gradient of the sigmoid function at
    #               each value of z (z can be a matrix, vector or scalar)
    tmp = sigmoid(z)
    g = tmp * (1 - tmp)
    # ===========================================
    return g

def randInitializeWeights(L_in, L_out):
    W = np.zeros((L_out, 1 + L_in))
    # ============= YOUR CODE HERE =============
    # Instructions: Initialize W randomly so that we break the symmetry while
    #               training the neural network.
    # Note: The first row of W corresponds to the parameters for the bias units
    epsilon_init = 0.12
    W = np.random.random((L_out, 1 + L_in)) * 2 * epsilon_init - epsilon_init
    # ===========================================
    return W
    
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_val):
    m = X.shape[0]
    J = 0
    Theta1, Theta2 = reshape_param(nn_params, input_layer_size, hidden_layer_size, num_labels)

    # ============= YOUR CODE HERE =============
    # Instructions:  Complete the following code to calculate the gradient function 
    #                by using feedforward and regularization
    X = np.column_stack((np.ones(m), X))
    z2_val = np.dot(X, np.transpose(Theta1))
    hiddenLayer = sigmoid(z2_val)
    hiddenLayer = np.column_stack((np.ones(m), hiddenLayer))
    outputLayer = sigmoid(np.dot(hiddenLayer, np.transpose(Theta2)))

    y_array = np.zeros((m, num_labels))
    for i in xrange(m):
        y_array[i, y[i]-1] = 1
    J1 = -y_array * np.log(outputLayer)
    J2 = (1 - y_array) * np.log(1 - outputLayer)
    J = np.sum(J1 - J2) / m
    J += np.sum(np.power(Theta1[:, 1:], 2)) * lambda_val / (2 * m)
    J += np.sum(np.power(Theta2[:, 1:], 2)) * lambda_val / (2 * m)
    # ===========================================
    return J

def nnGradFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_val):
    m = X.shape[0]
    Theta1, Theta2 = reshape_param(nn_params, input_layer_size, hidden_layer_size, num_labels)

    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # ============= YOUR CODE HERE =============
    # Instructions:  Complete the following code to calculate the gradient function 
    #                by using backpropagation and regularization
    X = np.column_stack((np.ones(m), X))
    z2_val = np.dot(X, np.transpose(Theta1))
    hiddenLayer = sigmoid(z2_val)
    hiddenLayer = np.column_stack((np.ones(m), hiddenLayer))
    outputLayer = sigmoid(np.dot(hiddenLayer, np.transpose(Theta2)))

    y_array = np.zeros((m, num_labels))
    for i in xrange(m):
        y_array[i, y[i]-1] = 1

    error_3 = outputLayer - y_array
    for t in xrange(m):
        error_3_col = error_3[t,:].reshape((-1,1))
        hiddenLayer_row = np.array([hiddenLayer[t, :]])
        z2_val_col = z2_val[t,:].reshape((-1,1))
        X_row = np.array([X[t,:]])

        Theta2_grad = Theta2_grad + np.dot(error_3_col, hiddenLayer_row)

        error_2 = np.dot(np.transpose(Theta2), error_3_col)
        error_2 = error_2[1:] # Remove bias term
        error_2 = error_2 * sigmoidGradient(z2_val_col)
        Theta1_grad = Theta1_grad + np.dot(error_2, X_row)

    Theta1_grad = Theta1_grad / m
    Theta2_grad = Theta2_grad / m

    Theta1_grad[:,1:] += Theta1[:,1:] * lambda_val / m
    Theta2_grad[:,1:] += Theta2[:,1:] * lambda_val / m
    # ===========================================
    grad = np.hstack((Theta1_grad.ravel(), Theta2_grad.ravel()))
    return grad

if __name__ == "__main__":
    plt.close('all')
    plt.ion() # interactive mode

    # Setup the parameters you will use for this part of the exercies
    input_layer_size = 400 # 20x20 Input Images of Digits
    hidden_layer_size = 25 # 25 hidden units
    num_labels = 10 # 10 labels, from 1 to 10   
    # (note that we have mapped "0" to label 10)

    # ==================== Part 1: Loading and Visualizing Data ====================
    
    print('Loading and Visualizing Data ...')

    data_file = '../../data/ex4/ex4data1.mat'
    mat_content = sio.loadmat(data_file)

    X = mat_content['X']
    y = mat_content['y']

    m, n = X.shape

    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]

    displayData(sel)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 2: Loading Parameters ===================

    print('Loading Saved Neural Network Parameters ...')

    data_file = '../../data/ex4/ex4weights.mat'
    mat_content = sio.loadmat(data_file)
    
    Theta1 = mat_content['Theta1']
    Theta2 = mat_content['Theta2']

    nn_params = np.hstack((Theta1.ravel(), Theta2.ravel()))

    # =================== Part 3: Compute Cost (Feedforward) ===================

    print('Feedforward Using Neural Network ...')

    # Weight regularization parameter (we set this to 0 here).
    lambda_val = 0

    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_val)

    print('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.287629)' % J)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 4: Implement Regularization ===================

    print('Checking Cost Function (w/ Regularization) ...')
    
    lambda_val = 1

    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_val)

    print('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.383770)' % J)
    
    raw_input('Program paused. Press enter to continue')

    # =================== Part 5: Sigmoid Gradient ===================
    
    print('Evaluating sigmoid gradient...')

    g = sigmoidGradient(np.array([1, -0.5, 0, 0.5, 1]))
    print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:')
    print(g)
    
    raw_input('Program paused. Press enter to continue')

    # =================== Part 6: Initializing Parameters ===================

    print('Initializing Neural Network Parameters ...')

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

    initial_nn_params = np.hstack((initial_Theta1.ravel(), initial_Theta2.ravel()))

    # =================== Part 7: Implement Backpropagation ===================
    
    print('Checking Backpropagation...')

    checkNNGradients(None)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 8: Implement Regularization ===================

    print('Checking Backpropagation (w/ Regularization) ...')

    lambda_val = 3
    checkNNGradients(lambda_val)

    debug_J  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_val)

    print('Cost at (fixed) debugging parameters (w/ lambda = 3): %f' % debug_J)
    print('(this value should be about 0.576051)')

    raw_input('Program paused. Press enter to continue')

    # =================== Part 9: Training NN ===================

    print('Training Neural Network...')

    lambda_val = 1
    costFunc = lambda p : nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_val)

    gradFunc = lambda p : nnGradFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_val)

    fmin_ret = optimize.fmin_cg(costFunc, initial_nn_params, gradFunc, maxiter=30, full_output=True)    

    nn_params = fmin_ret[0]
    cost = fmin_ret[1]

    print('Cost at theta found by fmin: %f' % cost)

    Theta1, Theta2 = reshape_param(nn_params, input_layer_size, hidden_layer_size, num_labels)

    raw_input('Program paused. Press enter to continue')
  
    # =================== Part 10: Visualize Weights ===================

    print('Visualizing Neural Network...')

    plt.figure()
    displayData(Theta1[:, 1:])

    raw_input('Program paused. Press enter to continue')

    # =================== Part 11: Implement Predict ===================

    pred = predict(Theta1, Theta2, X);

    print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))
