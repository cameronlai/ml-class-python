import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.neural_network import MLPClassifier
from ex4_utility import *

## Machine Learning Online Class - Exercise 4: Neural Network Learning

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  neural network exercise. You will need to complete a short section of code to perform logistic regression with scikit-learn library

#  Note that MLPClassifier is available in scikit-learn from 0.18dev0 or above
#  You may need to install the dev version in order to run this code

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

    # =================== Part 2: Finding Parameters ===================
    # Note that this parts does not load the suggested parameters,
    # but finds them using sci-kit learn

    print('Finding Neural Network Parameters ...')

    clf = None

    # ============= YOUR CODE HERE =============
    # Instructions: Use MLPClassifier to find the neural network parameters
    #               You may need to fine tine the parameter and settings to achieve good performance
    clf = MLPClassifier(hidden_layer_sizes=(hidden_layer_size), 
                        activation = 'logistic',
                        algorithm='sgd', 
                        alpha=0.1,
                        learning_rate_init=0.1,
                        max_iter=400,
                        random_state=1)
    clf.fit(X,y.ravel())
    # ===========================================

    if clf is None:
        sys.exit('Neural network model not initialized')
    else:
        print(clf)

    print('Neural network structure')
    print([coef.shape for coef in clf.coefs_])

    print('Number of iterations used: %f' % clf.n_iter_)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 3: Visualize Weights ===================

    print('Visualizing Neural Network...')

    plt.figure()
    displayData(np.transpose(clf.coefs_[0]))

    raw_input('Program paused. Press enter to continue')

    # =================== Part 4: Implement Predict ===================

    print('Training Set Accuracy: %f' % clf.score(X, y.ravel()));

    raw_input('Program paused. Press enter to continue')


