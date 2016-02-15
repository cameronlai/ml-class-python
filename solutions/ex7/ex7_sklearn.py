import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import scipy.io as sio
from sklearn.cluster import KMeans
from ex7 import findClosestCentroids, computeCentroids
from ex7_utility import plotProgresskMeans

## Machine Learning Online Class - Exercise 7: Principle Component Analysis and K-Means Clustering

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  clustering exercise. 
#  You will need to complete a short section of code to perform 
#  clustering with scikit-learn library
#

def runkMeans_sklearn(X, initial_centroids = None, max_iters= 0, plot_progress = False, input_K = 0):
    m, n = X.shape
    if initial_centroids is None:
        K = input_K
    else:
        K = initial_centroids.shape[0]
    idx = np.zeros((m,1))

    kmeans = None
    # ============= YOUR CODE HERE =============
    # Instructions: Perform K Means with sci-kit library
    #               Initialize with the given points
    #               If initial_centroids is an integer, then use random
    if initial_centroids is None:
        kmeans = KMeans(init='random', n_clusters=K, n_init=max_iters)
    else:
        kmeans = KMeans(init=initial_centroids, n_clusters=K, n_init=max_iters)
    kmeans.fit(X)
    # ===========================================
    if kmeans is None:
        sys.exit('K Means model not initialized')
        
    centroids = kmeans.cluster_centers_
    idx = kmeans.labels_
    
    if plot_progress:
        plotProgresskMeans(X, centroids, initial_centroids, idx, K, max_iters)

    return centroids, idx

if __name__ == "__main__":
    plt.close('all')
    plt.ion() # interactive mode

    # ==================== Part 1: Perform K Means ====================
    
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

    print('K-means starting point')
    plotProgresskMeans(X, initial_centroids, initial_centroids, idx, K, 0)

    raw_input('Press enter to continue')

    centroids, idx = runkMeans_sklearn(X, initial_centroids, max_iters, True)

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

    centroids, idx = runkMeans_sklearn(X, max_iters=max_iters, input_K = K)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 5: Image Compression ===================
    
    print('Applying K-Means to compress an image.')

    # Can use the idx trained from K Means instead of finding them again
    #idx = findClosestCentroids(X, centroids)

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
    plt.close('all')
