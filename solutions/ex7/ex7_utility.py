import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ex7 import *

def plotDataPoints(X, idx, K):
    colors = cm.rainbow(np.linspace(0, 1, K))
    for i in xrange(K):
        idx_loc = np.where(idx==i)[0]
        plt.scatter(X[idx_loc, 0], X[idx_loc, 1], marker='x', color=colors[i])

def plotProgresskMeans(X, centroids, previous, idx, K, i):
    plotDataPoints(X, idx, K)
    plt.scatter(centroids[:,0], centroids[:,1], marker='o', color='black')

    for j in xrange(K):
        tmp_centroids = np.vstack((previous[j, :], centroids[j,:]))
        plt.plot(tmp_centroids[:,0], tmp_centroids[:,1], color='black')

    plt.title('Iteration number %d' % (i))

def drawLine(p1, p2, formatString):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], formatString)

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = X - mu
    sigma = np.std(X, axis=0)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma

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

            display_array[row_pos:row_pos_end, col_pos:col_pos_end] = data / np.max(np.abs(data))
            
    display_array = np.transpose(display_array)
    plt.axis('off')
    plt.imshow(display_array, cmap=plt.cm.gray, vmin=-1, vmax=1)

