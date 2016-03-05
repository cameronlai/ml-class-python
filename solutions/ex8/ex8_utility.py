import numpy as np
import matplotlib.pyplot as plt
from ex8_cofi import *

def multivariateGaussian(X, mu, Sigma2):
    k = mu.size
    
    if Sigma2.shape[0] == 1 or Sigma2.shape[1] == 1:
        dim = np.max(Sigma2.shape)
        diag = Sigma2
        Sigma2 = np.zeros((dim, dim))
        np.fill_diagonal(Sigma2, diag)
        
    X = X - mu.transpose()
    prob = np.power((2 * np.pi), -k/2)
    prob *= np.power(np.linalg.det(Sigma2), -0.5)
    exp_power = -0.5 * X.dot(np.linalg.pinv(Sigma2)) * X
    prob *= np.exp(exp_power.sum(axis=1))
    return prob

def visualizeFit(X, mu, Sigma2):
    XGridVal = np.arange(0, 35, 0.5)
    X1, X2 = np.meshgrid(XGridVal, XGridVal)
    XVal = np.column_stack((X1.ravel(), X2.ravel()))

    Z = multivariateGaussian(XVal, mu, Sigma2)
    Z = Z.reshape(X1.shape)

    level = np.power(10.0, np.arange(-20, 0, 3))
    plt.contour(X1, X2, Z, level)

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

def checkCostFunction(lambda_val):
    if lambda_val is None:
        lambda_val = 0

    X_t = np.random.random((4,3))
    Theta_t = np.random.random((5,3))

    Y = X_t.dot(Theta_t.transpose())
    Y[np.random.random(Y.shape) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    X = np.random.random(X_t.shape)
    Theta = np.random.random(Theta_t.shape)
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    params = np.hstack((X.ravel(), Theta.ravel()))

    costFunc = lambda t : cofiCostFunc(t, Y, R, num_users, num_movies, num_features, lambda_val)

    gradFunc = lambda t : cofiGradFunc(t, Y, R, num_users, num_movies, num_features, lambda_val)

    cost = costFunc(params)
    grad = gradFunc(params)
    numgrad = computeNumericalGradient(costFunc, params)

    display_grad = np.column_stack((numgrad.ravel(), grad.ravel()))
    print(display_grad)
    print('The above two columns you get should be very similar')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)')

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If your backpropagation implementation is correct, then')
    print('the relative difference will be small (less than 1e-9).')
    print('Relative Difference: %f' % diff)

def loadMovieList():
    movieList = []
    data_file = '../../data/ex8/movie_ids.txt'
    f = open(data_file)
    n = 1682
    for i in xrange(n):
        line = f.readline()       
        idx, movieName = line.strip().split(' ', 1)
        movieList.append(movieName)
    f.close()
    return movieList

def normalizeRatings(Y, R):
    m, n = Y.shape
    Ymean = np.zeros((m,1))
    Ynorm = np.zeros(Y.shape)
    for i in xrange(m):
        idx = np.where(R[i, :] == 1)
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    return Ynorm, Ymean

def plot_datapoints(X):
    plt.plot(X[:,0], X[:,1], 'bx')
    plt.xlim([0,30])
    plt.ylim([0,30])
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')

def visualize_sklearn_clf(X, y_pred, threshold, clf, title = None):
    plot_datapoints(X)

    outliers = np.where(y_pred == 1)
    plt.plot(X[outliers, 0], X[outliers, 1], 'ro')

    XGridVal = np.arange(0, 35, 0.5)
    X1, X2 = np.meshgrid(XGridVal, XGridVal)
    Z = clf.decision_function(np.c_[X1.ravel(), X2.ravel()])
    Z = Z.reshape(X1.shape)
    plt.contourf(X1, X2, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Blues_r)

    if title is None:
        plt.title('Outlier detection with one class SVM')
    else:
        plt.title(title)



    
