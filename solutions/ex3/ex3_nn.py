import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import optimize
from ex3 import *

## Machine Learning Online Class - Exercise 3: Neural Network - One-vs-all

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     predict
#

# ==================== All function declaration ====================

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta1.shape[0]
    
    p = np.zeros((m, 1))    
    # ============= YOUR CODE HERE =============
    # Instructions:  Complete the following code to make predictions using
    #                your learned neural network. You should set p to a 
    #               vector containing labels between 1 to num_labels.
    X = np.column_stack((np.ones((m,1)), X))
    hidden = sigmoid(np.dot(X, np.transpose(Theta1)))
    hidden = np.column_stack((np.ones((m,1)), hidden))
    pre_output = sigmoid(np.dot(hidden, np.transpose(Theta2)))
    p = np.argmax(pre_output, axis=1) + 1
    p = p.reshape((-1, 1))
    # ===========================================
    return p

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

    data_file = '../../data/ex3/ex3data1.mat'
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

    data_file = '../../data/ex3/ex3weights.mat'
    mat_content = sio.loadmat(data_file)
    
    Theta1 = mat_content['Theta1']
    Theta2 = mat_content['Theta2']

    raw_input('Program paused. Press enter to continue')

    # =================== Part 3: Implement Predict ===================

    pred = predict(Theta1, Theta2, X)

    print('Training Set Accuracy: %f\n', np.mean(pred == y) * 100);

    raw_input('Program paused. Press enter to continue')

    rp = np.random.permutation(m)

    for i in rp:
        print('Displaying Example Image')
        data = X[i, :].reshape(1, -1)
        displayData(data)
        pred = predict(Theta1, Theta2, data)
        print('\nNeural Network Prediction: %d (digit %d)\n', pred[0][0], np.mod(pred, 10)[0][0]);

        raw_input('Program paused. Press enter to continue')
    


