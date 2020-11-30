import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error

data = np.loadtxt("lin_reg_train.txt")
#print(data.shape)
#print(data)
x_data_train = data[:, 0, np.newaxis]
y_data_train = data[:, 1, np.newaxis]
alpha=np.zeros(20)
x_pha = np.zeros((50,20))
for i in range(50):
    for j in range(20):
        alpha[j] = j*0.1-1
        x_pha[i,j] = np.exp(-5*np.power((x_data_train[i,0]-alpha[j]),2))
#print (x_pha)

#print(x_data_train.shape)
data_test = np.loadtxt("lin_reg_test.txt")
x_data_test = data_test[:, 0, np.newaxis]
y_data_test = data_test[:, 1, np.newaxis]
x_pha_test = np.zeros((100,20))
for i in range(100):
    for i in range(50):
        x_pha_test[i,j] = np.exp(-5*np.power((x_data_test[i,0]-alpha[j]),2))

#train_1 = np.r_[x_data_train[0:10],x_data_train[20:50]]
train_1 = x_data_train[10:50]
train_2 = x_data_train[0:10]
#y_train_1 = np.r_[y_data_train[0:10],y_data_train[20:50]]
y_train_1 = y_data_train[10:50]
y_train_2 = y_data_train[0:10]
#creat model, save error
lambda_to_test = 0.01


#1d
# n_samples, n_features = 50
# X = np.random.randn(n_samples, n_features)
# w = np.zeros(n_features)
model = BayesianRidge(alpha_1=0, lambda_1=0.01)
model.fit(x_pha,y_data_train.ravel())
pre_train = model.predict(x_pha)
pre_test = model.predict(x_pha_test)
error_train = np.sqrt(np.sum(np.power((y_data_train-pre_train),2))/50)
error_test = np.sqrt(np.sum((y_data_test-pre_test)**2)/100)
MLK_train = 50*np.log(1/np.sqrt(2*np.pi)/0.1)-np.sum(np.power((y_data_train-pre_train),2))/2/0.01
MLK_test = 100*np.log(1/np.sqrt(2*np.pi)/0.1)-np.sum(np.power((y_data_test-pre_test),2))/2/0.01
#print(MLK_test)
plt.scatter(x_data_train, y_data_train, marker='o',color='black', label = 'train_data')
# plt.scatter(x_data_test, y_data_test, marker='*',color='green', label = 'test_data')
plt.plot(x_data_train, pre_train, c= 'blue')
# plt.xlabel('x')
# plt.ylabel('y')
plt.show()