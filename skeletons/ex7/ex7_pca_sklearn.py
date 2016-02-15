import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from sklearn import decomposition
from ex7_sklearn import runkMeans_sklearn
from ex7_utility import *


## Machine Learning Online Class - Exercise 7: Principle Component Analysis and K-Means Clustering

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  PCA exercise. 
#  You will need to complete a short section of code to perform 
#  PCA with scikit-learn library
#

# ==================== All function declaration ====================

def pca(X, n_components):
    m, n = X.shape
    U = np.zeros((n,n))
    S = np.zeros((n,n))

    pca_model = None
    # ============= YOUR CODE HERE =============
    # Instructions: Use sklearn.decomposition.PCA to get principal components
    # ===========================================
    if pca is None:
        sys.exit('PCA is not initialized')

    U = -1 * pca_model.components_
    S = pca_model.explained_variance_

    return U, S, pca_model

if __name__ == "__main__":
    plt.close('all')
    plt.ion() # interactive mode

    # ==================== Part 1: Load Example Dataset ====================
    
    print('Visualizing example dataset for PCA.')

    data_file = '../../data/ex7/ex7data1.mat'
    mat_content = sio.loadmat(data_file)

    X = mat_content['X']

    plt.plot(X[:,0], X[:,1], 'bo')

    raw_input('Program paused. Press enter to continue')

    # =================== Part 2: Principal Component Analysis ===================

    print('Running PCA on example dataset.')

    X_norm, mu, sigma = featureNormalize(X);

    U, S, pca_model = pca(X_norm, 2)

    drawLine(mu, mu + 1.5 * S[0] * U[:,0], 'r-')
    drawLine(mu, mu + 1.5 * S[1] * U[:,1], 'g-')

    print('Top eigenvector: ')
    print(' U(:,1) = %f %f ' % (U[0,0], U[1,0]))
    print('(you should expect to see -0.707107 -0.707107)')

    raw_input('Program paused. Press enter to continue')

    # =================== Part 3: Dimension Reduction ===================

    print('Dimension reduction on example dataset.')

    plt.clf()
    plt.plot(X_norm[:,0], X_norm[:,1], 'bo')

    U, S, pca_model = pca(X_norm, 1)

    Z = pca_model.transform(X_norm)
    print('Projection of the first example: %f' % Z[0])
    print('(this value should be about 1.481274)')

    X_rec  = pca_model.inverse_transform(Z)
    print('Approximation of the first example: %f %f' % (X_rec[0,0], X_rec[0,1]))
    print('(this value should be about  -1.047419 -1.047419)')

    plt.plot(X_rec[:,0], X_rec[:,1], 'ro')
    for i in xrange(X_norm.shape[0]):
        drawLine(X_norm[i,:], X_rec[i,:], 'g--')

    raw_input('Program paused. Press enter to continue')

    # =================== Part 4: Loading and Visualizing Face Data ===================
    
    print('Loading face dataset.')

    data_file = '../../data/ex7/ex7faces.mat'
    mat_content = sio.loadmat(data_file)
    
    X = mat_content['X']

    plt.figure()
    displayData(X[:100,:])

    raw_input('Program paused. Press enter to continue')

    # =================== Part 5: PCA on Face Data: Eigenfaces ===================
    
    print('Running PCA on face dataset.\n(this mght take a minute or two ...)')

    X_norm, mu, sigma = featureNormalize(X)
    
    U, S, pca_model = pca(X_norm, n_components=100)

    plt.figure()
    displayData(U[:36,:])

    raw_input('Program paused. Press enter to continue')

    # =================== Part 6: Dimension Reduction for Faces ===================
    
    print('Dimension reduction for face dataset.')

    Z = pca_model.transform(X_norm)

    print('The projected data Z has a size of: ')
    print(Z.shape)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 7: Visualization of Faces after PCA Dimension Reduction ===================

    print('Visualizing the projected (reduced dimension) faces.')

    X_rec = pca_model.inverse_transform(Z)

    plt.subplot(1, 2, 1)
    displayData(X_norm[:100,:])
    plt.title('Original faces')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    displayData(X_rec[:100,:])
    plt.title('Recovered faces')
    plt.axis('off')

    raw_input('Program paused. Press enter to continue')

    # =================== Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===================
    
    plt.close('all')

    data_file = '../../data/ex7/bird_small.png'
    A = mpimg.imread(data_file)
    A = A / 255
    img_size = A.shape
    X = np.reshape(A, (img_size[0] * img_size[1], 3))

    K = 16
    max_iters = 10
    centroids, idx = runkMeans_sklearn(X, max_iters = max_iters, input_K=K)
    idx = idx.ravel()

    sel = np.floor(np.random.random(1000) * X.shape[0])
    sel = sel.astype('int')

    colors = cm.rainbow(np.linspace(0, 1, K))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[sel,0], X[sel,1], X[sel,2], c=colors[idx[sel], :])
    plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')

    raw_input('Program paused. Press enter to continue')

    # =================== Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===================

    X_norm, mu, sigma = featureNormalize(X)

    U, S, pca_model = pca(X_norm, n_components=2)
    Z = pca_model.transform(X_norm)

    plt.figure()
    plotDataPoints(Z[sel, :], idx[sel], K);
    plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');

    raw_input('Program paused. Press enter to continue')
    plt.close('all')

    
