import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import optimize
from ex8_utility import *

## Machine Learning Online Class - Exercise 8: Anomaly Detection and Collaborative Filtering

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions 
#  in this exericse:
#
#     cofiCostFunc
#     cofiGradFunc
#

# ==================== All function declaration ====================

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_val):
    X = np.reshape(params[:num_movies*num_features], (num_movies, num_features))
    Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features))

    J = 0
    # ============= YOUR CODE HERE =============
    # Instructions: Compute the cost function and gradient for collaborative
    #               filtering. Concretely, you should first implement the cost
    #               function (without regularization) and make sure it is
    #               matches our costs. After that, you should implement the 
    #               gradient and use the checkCostFunction routine to check
    #               that the gradient is correct. Finally, you should implement
    #               regularization.
    # ===========================================
    return J

def cofiGradFunc(params, Y, R, num_users, num_movies, num_features, lambda_val):
    X = np.reshape(params[:num_movies*num_features], (num_movies, num_features))
    Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features))

    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    # ============= YOUR CODE HERE =============
    # Instructions: Compute the cost function and gradient for collaborative
    #               filtering. Concretely, you should first implement the cost
    #               function (without regularization) and make sure it is
    #               matches our costs. After that, you should implement the 
    #               gradient and use the checkCostFunction routine to check
    #               that the gradient is correct. Finally, you should implement
    #               regularization.
    # ===========================================
    grad = np.hstack((X_grad.ravel(), Theta_grad.ravel()))
    return grad

if __name__ == "__main__":
    plt.close('all')
    plt.ion() # interactive mode

    # ==================== Part 1: Load movie ratings dataset ====================
    
    print('Loading movie ratings dataset.')

    data_file = '../../data/ex8/ex8_movies.mat'
    mat_content = sio.loadmat(data_file)

    Y = mat_content['Y']
    R = mat_content['R']

    plt.imshow(Y)
    plt.ylabel('Movies')
    plt.xlabel('Users')

    raw_input('Program paused. Press enter to continue')

    # =================== Part 2: Collaborative Filtering Cost Function ===================

    data_file = '../../data/ex8/ex8_movieParams.mat'
    mat_content = sio.loadmat(data_file)

    X = mat_content['X']
    Theta = mat_content['Theta']

    # Reduce the data set size so that this runs faster
    num_users = 4
    num_movies = 5
    num_features = 3

    X = X[:num_movies, :num_features]
    Theta = Theta[:num_users, :num_features]
    Y = Y[:num_movies, :num_users]
    R = R[:num_movies, :num_users]

    params = np.hstack((X.ravel(), Theta.ravel()))
    J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 0)
    
    print('Cost at loaded parameters: %f' % J)
    print('(this value should be about 22.22)')


    raw_input('Program paused. Press enter to continue')

    # =================== Part 3: Collaborative Filtering Gradient ===================

    print('Checking Gradients (without regularization) ... ')
    
    checkCostFunction(0)
    
    raw_input('Program paused. Press enter to continue')

    # =================== Part 4: Collaborative Filtering Cost Regularization ===================
    
    params = np.hstack((X.ravel(), Theta.ravel()))
    J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 1.5)
           
    print('Cost at loaded parameters (lambda = 1.5): %f' % J)
    print('this value should be about 31.34)')
    
    raw_input('Program paused. Press enter to continue')

    # =================== Part 5: Collaborative Filtering Gradient Regularization ===================

    print('Checking Gradients (with regularization) ... ')
    
    checkCostFunction(1.5)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 6: Entering ratings for a new user ===================

    movieList = loadMovieList()

    my_ratings = np.zeros((1682, 1))

    my_ratings[0] = 4
    my_ratings[97] = 2

    my_ratings[6] = 3;
    my_ratings[11]= 5;
    my_ratings[53] = 4;
    my_ratings[63]= 5;
    my_ratings[65]= 3;
    my_ratings[68] = 5;
    my_ratings[182] = 4;
    my_ratings[225] = 5;
    my_ratings[354]= 5;

    print('New user ratings')
    for i in xrange(len(my_ratings)):
        if my_ratings[i] > 0:
            print('Rated %d for %s' % (my_ratings[i], movieList[i]))

    raw_input('Program paused. Press enter to continue')

    # =================== Part 7: Learning Movie Ratings ===================

    print('Training collaborative filtering...')

    data_file = '../../data/ex8/ex8_movies.mat'
    mat_content = sio.loadmat(data_file)

    Y = mat_content['Y']
    R = mat_content['R']

    Y = np.column_stack((my_ratings, Y))
    R = np.column_stack(((my_ratings != 0), R))
    
    Ynorm, Ymean = normalizeRatings(Y, R)

    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = 10

    X = np.random.random((num_movies, num_features))
    Theta = np.random.random((num_users, num_features))

    initial_parameters = np.hstack((X.ravel(), Theta.ravel()))
    
    lambda_val = 10

    costFunc = lambda t : cofiCostFunc(t, Y, R, num_users, num_movies, num_features, lambda_val)
    gradFunc = lambda t : cofiGradFunc(t, Y, R, num_users, num_movies, num_features, lambda_val)

    fmin_ret = optimize.fmin_cg(costFunc, initial_parameters, gradFunc, maxiter=100, full_output=True)

    params = fmin_ret[0]
    X = np.reshape(params[:num_movies*num_features], (num_movies, num_features))
    Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features))

    print('Recommender system learning completed.')
    
    raw_input('Program paused. Press enter to continue')

    # =================== Part 8: Recommendation for you ===================

    p = X.dot(Theta.transpose())
    my_predictions = p[:,0] + Ymean.ravel()

    idx = np.argsort(my_predictions)[::-1]
    
    print('Top recommendations for you:')
    for i in xrange(10):
        j = idx[i]
        print('Predicting rating %.1f for movie %s' %  (my_predictions[j], movieList[j]))

    print('\nOriginal ratings provided:');
    for i in xrange(10):
        if my_ratings[i] > 0:
            print('Rated %d for %s\n' % (my_ratings[i], movieList[i]))

    raw_input('Program paused. Press enter to continue')    
    plt.close('all')
