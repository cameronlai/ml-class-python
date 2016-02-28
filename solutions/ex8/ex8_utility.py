import numpy as np
import matplotlib.pyplot as plt

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



