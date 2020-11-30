import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
data = data = np.loadtxt("lin_reg_train.txt")
x_data_train = data[:, 0, np.newaxis]
y_data_train = data[:, 1, np.newaxis]
data_test = np.loadtxt("lin_reg_test.txt")
x_data_test = data_test[:, 0, np.newaxis]
y_data_test = data_test[:, 1, np.newaxis]

n_samples, n_features = 50
X = np.random.randn(n_samples, n_features)
w = np.zeros(n_features,2)
lambda_I = 0.01*np.eye(50,dtype=int)
alpha_ = 5
noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=n_samples)
for i in range(50):
    w[i,0] = stats.norm.rvs(loc=0, scale=0.01)
    w[i,1] = noise

def BayesLR():
