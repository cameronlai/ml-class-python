import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import scipy.io as sio
from ex7_utility import *

## Machine Learning Online Class - Exercise 7: Principle Component Analysis and K-Means Clustering

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions 
#  in this exericse:
#
#     computeCentroids
#     findClosestCentroids
#     kMeansInitCentroids
#

# ==================== All function declaration ====================

def findClosestCentroids(X, centroids):
    m = X.shape[0]
    K = centroids.shape[0]
    idx = np.zeros((m,1), dtype=int)
    # ============= YOUR CODE HERE =============
    # Instructions: Go over every example, find its closest centroid, and store
    #               the index inside idx at the appropriate location.
    #               Concretely, idx(i) should contain the index of the centroid
    #               closest to example i. Hence, it should be a value in the 
    #               range 1..K
    # ===========================================
    return idx

def computeCentroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    # ============= YOUR CODE HERE =============
    # Instructions: Go over every centroid and compute mean of all points that
    #               belong to it. Concretely, the row vector centroids(i, :)
    #               should contain the mean of the data points assigned to
    #               centroid i.
    # ===========================================
    return centroids

def runkMeans(X, initial_centroids, max_iters, plot_progress = False):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m,1))

    for i in xrange(max_iters):
        print('K-Means iteration %d/%d...\n' % (i+1, max_iters))
        idx = findClosestCentroids(X, centroids)
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            raw_input('Press enter to continue')
        
        centroids = computeCentroids(X, idx, K)
    return centroids, idx

def kMeansInitCentroids(X, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    # ============= YOUR CODE HERE =============
    # Instructions: You should set centroids to randomly chosen examples from
    #               the dataset X
    # ===========================================
    return centroids

if __name__ == "__main__":
    plt.close('all')
    plt.ion() # interactive mode

    # ==================== Part 1: Find Closest Centroids ====================
    
    print('Finding closest centroids.')

    data_file = '../../data/ex7/ex7data2.mat'
    mat_content = sio.loadmat(data_file)

    X = mat_content['X']

    K = 3
    initial_centroids = np.array([[3,3], [6,2], [8,5]])

    idx = findClosestCentroids(X, initial_centroids)
    
    print('Closest centroids for the first 3 examples:')
    print(idx[:3].ravel());
    print('(the closest centroids should be 0, 2, 1 respectively)')

    raw_input('Program paused. Press enter to continue')

    # =================== Part 2: Compute Means ===================

    print('Computing centroids means.')

    centroids = computeCentroids(X, idx, K);

    print('Centroids computed after initial finding of closest centroids: ')
    print(centroids);
    print('(the centroids should be');
    print('   [ 2.428301 3.157924 ]');
    print('   [ 5.813503 2.633656 ]');
    print('   [ 7.119387 3.616684 ]');

    raw_input('Program paused. Press enter to continue')

    # =================== Part 3: K-Means Clustering ===================

    print('Running K-Means clustering on example dataset.')

    max_iters = 10

    centroids, idx = runkMeans(X, initial_centroids, max_iters, True)
    print('K-Means Done.')

    raw_input('Program paused. Press enter to continue')

    # =================== Part 4: K-Means Clustering on Pixels ===================
    
    print('Running K-Means clustering on pixels from an image.')

    data_file = '../../data/ex7/bird_small.png'
    A = mpimg.imread(data_file)
    A = A / 255
    img_size = A.shape

    X = np.reshape(A, (img_size[0] * img_size[1], 3))

    K = 16
    max_iters = 10

    initial_centroids = kMeansInitCentroids(X, K)

    centroids, idx = runkMeans(X, initial_centroids, max_iters)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 5: Image Compression ===================
    
    print('Applying K-Means to compress an image.')

    idx = findClosestCentroids(X, centroids)

    X_recovered = centroids[idx,:]
    
    X_recovered = np.reshape(X_recovered, (img_size[0], img_size[1], 3))

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(A * 255)
    plt.title('Original')

    plt.subplot(1,2,2)
    plt.imshow(X_recovered * 255)
    plt.title('Compressed, with %d colors.' % K)

    raw_input('Program paused. Press enter to continue')
