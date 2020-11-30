import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from scipy.linalg import cho_solve
from numpy.linalg import cholesky
from itertools import cycle


def _exponential_cov(x1, x2,theta):
    return theta[0] * np.exp(-0.5*theta[1]* np.subtract.outer(x1, x2)** 2 )

def conditional(x_new, x, y, theta):
    A = _exponential_cov(x_new, x, theta)
    B = _exponential_cov(x, x, theta)
    C = _exponential_cov(x_new, x_new, theta)
    mu = np.linalg.inv(B).dot(A.T).T.dot(y)
    sigma = C-A.dot(np.linalg.inv(B).dot(A.T))
    return np.squeeze(mu), np.squeeze(sigma)

theta = [1,1]
delta_0 = _exponential_cov(0, 0, theta)
x = [1]
y = [np.random.normal(scale=delta_0)]
delta_1 = _exponential_cov(x, x, theta)
def predict(x,data,kernel,theta, sigma, t):
    k = [kernel(x, y, theta) for y in data]
    sinv = np.linalg.inv(sigma)
    y_pre = np.dot(k, sinv).dot(t)
    sigma_new = kernel(x, x,theta)-np.dot(k, sinv).dot(k)
    return y_pre, sigma_new
x_pre = np.arange(0,2*np.pi,0.005)
y_pre = np.sin(x_pre)+np.sin(x_pre)**2
predictions = [predict(i, x, _exponential_cov, theta, delta_1, y) for i in x_pre]
y_pre, sigmas = np.transpose(predictions)
plt.errorbar(x_pre, y_pre, yerr=sigmas, capsize=0)
plt.plot(x, y, "ro")

plt.show()