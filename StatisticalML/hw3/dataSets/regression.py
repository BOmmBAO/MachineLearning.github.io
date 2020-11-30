import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures   #generet polynimie
# from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt("lin_reg_train.txt")
x_data = data[:,0]
y_data = data[:,1]
#x__data = np.concatenate((np.ones((100,1)),x_data), axis  = 1)  concatenate matrix
plt.scatter(x_data, y_data)
plt.show()
# x_data = data[:,0,np.newaxis]#add a dimention
# y_data = data[:,1,np.newaxis]
# model = LinearRegression()
# model.fit(x_data, y_data)
#
#model = linear_model.LinearRegression()#实例化
# plt.plot(x_data, y_data, "b")
# plt.plot(x_data, model.predict(x_data), "r")
# plt.show()

# ax = plt.figure().add_subplot(111, projection = "3d")
# ax.scatter(x,y,z, c = "r",, maker = "o", s = 100)#point is red triangle
# x0 =
# x1 =
# x0,x1 = np.meshgrid(x0,x1)
# ax.plot_surface(x0, x1, z)
# ax.set_xlabel("difhid")