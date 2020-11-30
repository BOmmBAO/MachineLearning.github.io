import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error

data = np.loadtxt("lin_reg_train.txt")
#print(data.shape)
#print(data)
x_data_train = data[:, 0, np.newaxis]
y_data_train = data[:, 1, np.newaxis]
#print(x_data)
data_test = np.loadtxt("lin_reg_test.txt")
x_data_test = data_test[:, 0, np.newaxis]
y_data_test = data_test[:, 1, np.newaxis]

#creat model, save error
lambda_to_test = 0.01
"""1a
model = linear_model.Ridge(alpha=lambda_to_test,fit_intercept=False)
model.fit(x_data_train, y_data_train)
# ridge coefficient
pre = model.predict(x_data_test)
#print(pre)
#print(data_test.shape)

#loss value
print(np.sqrt(np.sum((y_data_test-pre)**2)/100))

plt.scatter(x_data_train, y_data_train, marker='o',color='black', label = 'train_data')
plt.scatter(x_data_test, y_data_test, marker='*',color='green', label = 'test_data')
plt.plot(x_data_test, pre, c= 'blue')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""
#polynomial
"""
i = 4
poly_reg = PolynomialFeatures(degree=i)
x_poly = poly_reg.fit_transform(x_data_train)
model = linear_model.Ridge(alpha=lambda_to_test)
model.fit(x_poly, y_data_train)
x_poly_test = poly_reg.fit_transform(x_data_test)
result_train = model.predict(x_poly)
pre = model.predict(x_poly_test)
error = np.sqrt(np.sum((y_data_test-pre)**2)/100)
print(error)
plt.scatter(x_data_train, y_data_train, marker='o',color='black', label = 'train_data')
plt.scatter(x_data_test, y_data_test, marker='*',color='green', label = 'test_data')
plt.plot(x_data_train, result_train, c= 'blue')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

"""
#1c K Fold cross validation
# skf = StratifiedKFold( n_splits=5, shuffle=False)
# skf.get_n_splits(x_data_train, y_data_train)
# model  = linear_model.Ridge(alpha=lambda_to_test)
# print(skf)
# for train_1_index, train_2_index, train_3_index, train_4_index, train_5_index in skf.split(x_data_train, y_data_train):
#    print("TRAIN:", train_index, "TEST:", valid_index)
#    x_1, x_2, x_3, x_4, x_5 = x_data_train[train_1_index], x_data_train[train_2_index], x_data_train[train_3_index], x_data_train[train_4_index], x_data_train[train_5_index]
#    y_1, y_2, y_3, y_4, y_5 = y_data_train[train_1_index], y_data_train[train_2_index], y_data_train[train_3_index], \
#                              y_data_train[train_4_index], y_data_train[train_5_index]
# #
# # for deg in range(2,5):
# #     x_train, x_valid, y_train, y_valid = train_test_split(x_data_train, y_data_train, test_size=0.2)
# #     poly = PolynomialFeatures(deg)
# #     x_train_poly = poly.fit_transform(x_train)
# #     x_valid_poly = poly.fit_transform(x_valid)
# #
# #     model.fit(x_poly, y_data_train)
"""
