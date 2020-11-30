import numpy as np
import matplotlib.pyplot as plt
# from sklearn.cross_validation import KFold

lin_reg_train = np.loadtxt("lin_reg_train.txt")
lin_reg_test = np.loadtxt("lin_reg_test.txt")


# 1a
def linear_features(data, lam=0.01):
    x_array = np.c_[np.ones(data[:, 0].shape[0]), data[:, 0]]
    y_array = data[:, 1]
    x = np.mat(x_array)
    y = np.mat(y_array).reshape(-1, 1)
    w = np.linalg.inv(x.T * x + lam * np.eye(x.shape[1])) * x.T * y
    return x, y, w


def quadratic_features(data, lam=0.01):
    x0 = data[:, 0]
    x_array = np.c_[np.ones(x0.shape[0]), x0, np.square(x0)]
    y_array = data[:, 1]
    x_p = np.linspace(np.min(x_array.ravel()), np.max(x_array.ravel()), 10000).reshape((-1, 1))
    x_plot = np.c_[np.ones(x_p.shape[0]), x_p, np.square(x_p)]
    x = np.mat(x_array)
    y = np.mat(y_array).reshape(-1, 1)
    w = np.linalg.inv(x.T * x + lam * np.eye(x.shape[1])) * x.T * y
    return x_p, x_plot, x, y, w


def cubic_features(data, lam=0.01):

    x0 = data[:, 0]
    x_array = np.c_[np.ones(x0.shape[0]), x0, np.square(x0), np.power(x0, 3)]
    y_array = data[:, 1]
    x_p = np.linspace(np.min(x_array.ravel()), np.max(x_array.ravel()), 10000).reshape((-1, 1))
    x_plot = np.c_[np.ones(x_p.shape[0]), x_p, np.square(x_p), np.power(x_p, 3)]
    x = np.mat(x_array)
    y = np.mat(y_array).reshape(-1, 1)
    w = np.linalg.inv(x.T * x + lam * np.eye(x.shape[1])) * x.T * y
    return x_p, x_plot, x, y, w


def quartic_features(data, lam=0.01):

    x0 = data[:, 0]
    x_array = np.c_[np.ones(x0.shape[0]), x0, np.square(x0), np.power(x0, 3), np.power(x0, 4)]
    y_array = data[:, 1]
    x_p = np.linspace(np.min(x_array.ravel()), np.max(x_array.ravel()), 10000).reshape((-1, 1))
    x_plot = np.c_[np.ones(x_p.shape[0]), x_p, np.square(x_p), np.power(x_p, 3), np.power(x_p, 4)]
    x = np.mat(x_array)
    y = np.mat(y_array).reshape(-1, 1)
    w = np.linalg.inv(x.T * x + lam * np.eye(x.shape[1])) * x.T * y
    return x_p, x_plot, x, y, w


def RMSE(target, predict):
    return np.sqrt(np.mean(np.square(predict-target)))


x_train, y_train, w_train = linear_features(lin_reg_train)
EMSE_train = RMSE(y_train, x_train*w_train)
x_test, y_test, w_test = linear_features(lin_reg_test)
EMSE_test = RMSE(y_test, x_test*w_test)
print('linear features:')
print('EMSE of train data:', EMSE_train)
print('EMSE of test data:', EMSE_test)

plt.scatter(lin_reg_train[:, 0], lin_reg_train[:, 1], c='black', label='train data')
plt.plot(lin_reg_train[:, 0], x_train*w_train, c='blue', label='predicted function')
plt.title('linear features')
plt.legend()
plt.show()


# 1b
x_plot_raw_train, x_plot_train, x_train, y_train, w_train = quadratic_features(lin_reg_train)
EMSE_train = RMSE(y_train, x_train*w_train)
x_plot_raw_test,x_plot_test, x_test, y_test, w_test = quadratic_features(lin_reg_test)
EMSE_test = RMSE(y_test, x_test*w_test)
print('polynomials of degrees 2:')
print('EMSE of train data:', EMSE_train)
print('EMSE of test data:', EMSE_test)
plt.scatter(lin_reg_train[:, 0], lin_reg_train[:, 1], c='black', label='train data')
plt.plot(x_plot_raw_train, x_plot_train * w_train, c='blue', label='predicted function')
plt.title('quadratic_features')
plt.legend()
plt.show()

x_plot_raw_train, x_plot_train, x_train, y_train, w_train = cubic_features(lin_reg_train)
EMSE_train = RMSE(y_train, x_train*w_train)
x_plot_raw_test,x_plot_test, x_test, y_test, w_test = cubic_features(lin_reg_test)
EMSE_test = RMSE(y_test, x_test*w_test)
print('polynomials of degrees 3:')
print('EMSE of train data:', EMSE_train)
print('EMSE of test data:', EMSE_test)
plt.scatter(lin_reg_train[:, 0], lin_reg_train[:, 1], c='black', label='train data')
plt.plot(x_plot_raw_train, x_plot_train * w_train, c='blue', label='predicted function')
plt.title('cubic_features')
plt.legend()
plt.show()

x_plot_raw_train, x_plot_train, x_train, y_train, w_train = quartic_features(lin_reg_train)
EMSE_train = RMSE(y_train, x_train*w_train)
x_plot_raw_test,x_plot_test, x_test, y_test, w_test = quartic_features(lin_reg_test)
EMSE_test = RMSE(y_test, x_test*w_test)
print('polynomials of degrees 4:')
print('EMSE of train data:', EMSE_train)
print('EMSE of test data:', EMSE_test)
plt.scatter(lin_reg_train[:, 0], lin_reg_train[:, 1], c='black', label='train data')
plt.plot(x_plot_raw_train, x_plot_train * w_train, c='blue', label='predicted function')
plt.title('quartic_features')
plt.legend()
plt.show()




