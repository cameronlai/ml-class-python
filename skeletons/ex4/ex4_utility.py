import numpy as np
from ex4 import *

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta.shape[0]    
    p = np.zeros((m,1))
    
def displayData(X):
    m, n = X.shape
    example_width = int(np.round(np.sqrt(n)))
    example_height = int(n / example_width)

    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    pad = 1
    display_array = np.ones((pad + display_rows * (example_height + pad),
                             pad + display_cols * (example_width + pad)))

    for j in range(display_rows):
        for i in range(display_cols):
            row_pos = pad + (example_height + pad) * j
            col_pos = pad + (example_width + pad) * i

            row_pos_end = row_pos + example_height
            col_pos_end = col_pos + example_width
            
            data = X[j*display_rows + i, :]
            data = data.reshape((example_height, example_width))

            display_array[row_pos:row_pos_end, col_pos:col_pos_end] = data / np.max(data)
            
    display_array = np.transpose(display_array)
    plt.imshow(display_array, cmap=plt.cm.gray)

def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    W = np.reshape(np.sin(np.arange(W.size)) / 10, W.shape)
    return W

def computeNumericalGradient(J, theta):
    numgrad = np.zeros(theta.size)
    perturb = np.zeros(theta.size)
    e = 1e-4
    for p in range(theta.size):
        perturb[p] = e
        tmpPerturb = perturb.reshape(theta.shape)
        loss1 = J(theta - tmpPerturb)
        loss2 = J(theta + tmpPerturb)
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0       
    numgrad = numgrad.reshape(theta.shape)
    return numgrad

def checkNNGradients(lambda_val):
    if lambda_val is None:
        lambda_val = 0

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # Random data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m, input_layer_size - 1);
    y  = np.transpose(1 + np.mod(np.arange(m), num_labels))

    nn_params = np.hstack((Theta1.ravel(), Theta2.ravel()))

    costFunc = lambda p : nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_val)

    gradFunc = lambda p : nnGradFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_val)

    cost = costFunc(nn_params) 
    grad = gradFunc(nn_params) 
    numgrad = computeNumericalGradient(costFunc, nn_params)

    display_grad = np.column_stack((numgrad.ravel(), grad.ravel()))
    print(display_grad)
    print('The above two columns you get should be very similar')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)')

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If your backpropagation implementation is correct, then')
    print('the relative difference will be small (less than 1e-9).')
    print('Relative Difference: %f' % diff)
