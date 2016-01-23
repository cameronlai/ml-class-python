import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import optimize

## Machine Learning Online Class - Exercise 3: Neural Network - One-vs-all

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     lrCostFunction (logistic regression cost function)
#     lrGradFucntion (logistic regression partial derivative function)
#     oneVsAll
#     predictOneVsAll
#

# ==================== All function declaration ====================

def sigmoid(z):
    g = np.zeros(z.shape)
    g = 1 / (1 + np.exp(-z))
    return g

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

def lrCostFunction(theta, X, y, lambda_val):
    m = y.shape[0]
    J = 0
    # ============= YOUR CODE HERE =============
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    h_val= sigmoid(np.dot(X, theta)).reshape((-1, 1))
    J = np.sum(-y * np.log(h_val) - (1-y) * np.log(1-h_val)) / m
    J = J + lambda_val * np.sum(np.power(theta[1:], 2)) / (2 * m)
    # ===========================================
    return J

def lrGradFunction(theta, X, y, lambda_val):
    m = y.shape[0]
    grad = np.zeros(theta.shape)
    # ============= YOUR CODE HERE =============
    # Instructions: Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    h_val= sigmoid(np.dot(X, theta)).reshape((-1, 1))
    grad = np.sum((h_val-y) * X, axis=0) / m
    grad[1:] = grad[1:] + lambda_val * theta[1:] / m
    # ===========================================
    return grad

def oneVsAll(X, y, num_lables, lambda_val):
    m, n = X.shape
    X = np.column_stack((np.ones((m, 1)), X))
    all_theta = np.zeros((num_labels, n+1))
    for c in range(num_labels):
        # ============= YOUR CODE HERE =============
        # Instructions: You should complete the following code to train num_labels 
        #               logistic regression classifiers with regularization
        #               parameter lambda.

        initial_theta = np.zeros((n+1, 1))
        label_y = np.where(y==(c+1), 1, 0 )
        fmin_ret = optimize.fmin_cg(lambda t : lrCostFunction(t, X, label_y, lambda_val), 
                                    initial_theta,
                                    lambda t : lrGradFunction(t, X, label_y, lambda_val), 
                                     maxiter=50, full_output = True)
        theta = fmin_ret[0]
        cost = fmin_ret[1]
        all_theta[c, :] = theta        
        # ===========================================
        print('Label %d: Cost found is %f' % (c+1, cost))
    return all_theta

def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    
    p = np.zeros((m, 1))    
    X = np.hstack((np.ones((m, 1)), X))
    # ============= YOUR CODE HERE =============
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters (one-vs-all).
    #               You should set p to a vector of predictions (from 1 to
    #               num_labels).
    y_est = np.dot(X, np.transpose(all_theta))
    p = np.argmax(y_est, axis=1) + 1
    p = p.reshape((-1, 1))
    # ===========================================
    return p

if __name__ == "__main__":
    plt.close('all')
    plt.ion() # interactive mode

    # Setup the parameters you will use for this part of the exercies
    input_layer_size = 400 # 20x20 Input Images of Digits
    num_labels = 10 # 10 labels, from 1 to 10   
    # (note that we have mapped "0" to label 10)

    # ==================== Part 1: Loading and Visualizing Data ====================
    
    print('Loading and Visualizing Data ...')

    data_file = '../../data/ex3/ex3data1.mat'
    mat_content = sio.loadmat(data_file)

    X = mat_content['X']
    y = mat_content['y']

    m, n = X.shape

    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]

    displayData(sel)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 2: Vectorize Logistic Regression ===================

    print('Training One-vs-All Logistic Regression...')

    lambda_val = 0.1
    all_theta = oneVsAll(X, y, num_labels, lambda_val)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 3: Vectorize Logistic Regression ===================

    pred = predictOneVsAll(all_theta, X)

    print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))
