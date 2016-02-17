import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import svm

## Machine Learning Online Class - Exercise 6: Support Vector Machines

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     gaussianKernel
#     dataset3Params
#

# ==================== All function declaration ====================

def plotData(X, y):
    pos = np.where(y==1)[0]
    neg = np.where(y==0)[0]
    plt.plot(X[pos, 0], X[pos, 1], 'b+', label='Training data')
    plt.plot(X[neg, 0], X[neg, 1], 'ro', label='Training data')
    plt.legend()

def visualizeBoundaryLinear(X, y, model):
    w = model.coef_[0]
    b = model.intercept_
    xp = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    yp = -(w[0] * xp + b) / w[1]     # Boundary is at y=0
    plt.plot(xp, yp, 'b-', label='Decision boundary')
    plt.legend()

def visualizeBoundary(X, y, model):
    x1vals = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    x2vals = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)
    xv, yv = np.meshgrid(x1vals, x2vals)
    zv = model.predict(np.c_[xv.ravel(), yv.ravel()])
    zv = zv.reshape(xv.shape)
    plt.contour(xv, yv, zv, [0,0], colors='blue', label='Decision Boundary')
    plt.legend()

def sklearnGaussianKernel(X, Y, sigma):
    m = X.shape[0]
    n = Y.shape[0]
    gram_matrix = np.zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            gram_matrix[i, j] = gaussianKernel(X[i,:], Y[j,:], sigma)
    return gram_matrix

def gaussianKernel(x1, x2, sigma):
    sim = 0
    # ============= YOUR CODE HERE =============
    # Instructions: Fill in this function to return the similarity between x1
    #               and x2 computed using a Gaussian kernel with bandwidth
    #               sigma
    dis = np.sum(np.power(x1-x2, 2))
    sim = np.exp(-dis/(2*np.power(sigma, 2)))
    # ===========================================
    return np.array([[sim]])

def dataset3Params(X, y, Xval, yval):
    C = 1
    sigma = 0.3
    # ============= YOUR CODE HERE =============
    # Instructions: Fill in this function to return the optimal C and sigma
    #               learning parameters found using the cross validation set.
    #               You can use model.predict to predict the labels on the cross
    #               validation set. 
    test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    max_score = 0
    for i in xrange(8):
        for j in xrange(8):
            tC = test[i]
            tSigma = test[j]

            model = svm.SVC(C=tC, kernel='rbf', gamma=1.0/tSigma, max_iter=200)
            model.fit(X, y)
            score = model.score(Xval, yval)
            print(score)
            if score > max_score:
                max_score = score
                C = tC
                sigma = tSigma
    # ===========================================
    return C, sigma

if __name__ == "__main__":
    plt.close('all')
    plt.ion() # interactive mode

    # ==================== Part 1: Loading and Visualizing Data ====================
    
    print('Loading and Visualizing Data ...')

    data_file = '../../data/ex6/ex6data1.mat'
    mat_content = sio.loadmat(data_file)

    X = mat_content['X']
    y = mat_content['y']

    m, n = X.shape
    plt.figure()
    plotData(X, y)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 2: Training Linear SVM ===================

    C = 1
    model = svm.SVC(C=C, kernel='linear', max_iter=20)
    model.fit(X, y)
    visualizeBoundaryLinear(X, y, model)
    
    raw_input('Program paused. Press enter to continue')

    # =================== Part 3: Implementing Gaussian Kernel ===================

    print('Evaluating the Gaussian Kernel ...')
    
    x1 = np.array([1, 2, 1], dtype='f')
    x2 = np.array([0, 4, -1], dtype='f')
    sigma = 2
    sim = gaussianKernel(x1, x2, sigma)

    print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5 : %f\n(this value should be about 0.324652)' % sim);
    
    raw_input('Program paused. Press enter to continue')

    # =================== Part 4: Visualizing Dataset 2 ===================
    
    print('Loading and Visualizing Data ...')
    
    data_file = '../../data/ex6/ex6data2.mat'
    mat_content = sio.loadmat(data_file)

    X = mat_content['X']
    y = mat_content['y']

    m, n = X.shape
    plt.figure()
    plotData(X, y)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 5: Training SVM with RBF Kernel (Dataset 2) ===================
    
    print('Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...')

    C = 1.0
    sigma = 0.1
    
    # Use kernel function, but will be very slow
    #kernel_func = lambda X, Y: sklearnGaussianKernel(X, Y, sigma)
    #model = svm.SVC(C=C, kernel=kernel_func, max_iter=200)
    
    # Use libSVM's RBF kernel
    model = svm.SVC(C=C, kernel='rbf', gamma=1/sigma, max_iter=200)

    model.fit(X, y)

    print('Finish training, now draw contour decision boundary')

    visualizeBoundary(X, y, model)

    raw_input('Program paused. Press enter to continue')

    # =================== Part 6: Visualizing Dataset 3 ===================

    print('Loading and Visualizing Data ...')

    data_file = '../../data/ex6/ex6data3.mat'
    mat_content = sio.loadmat(data_file)

    X = mat_content['X']
    y = mat_content['y']
    Xval = mat_content['Xval']
    yval = mat_content['yval']

    m, n = X.shape
    plt.figure()
    plotData(X, y)

    raw_input('Program paused. Press enter to continue')
    
    # =================== Part 7: Training SVM with RBF Kernel (Dataset 3 ===================

    C, sigma = dataset3Params(X, y, Xval, yval)

    print('C found is = %f' % C)
    print('sigma found is = %f' % sigma)

    # Use kernel function, but will be very slow
    #kernel_func = lambda X, Y: sklearnGaussianKernel(X, Y, sigma)
    #model = svm.SVC(C=C, kernel=kernel_func, max_iter=20, verbose=True)

    # Use libSVM's RBF kernel
    model = svm.SVC(C=C, kernel='rbf', gamma=1/sigma, max_iter=200)

    model.fit(X, y)
    
    visualizeBoundary(X, y, model)

    raw_input('Program paused. Press enter to continue')
    plt.close('all')
    
    
    
    


