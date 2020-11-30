from sklearn import  discriminant_analysis

def plot_LDA(converted_X, y):
    '''
    draw the LDA transformed data
    :param converted_X: data set after Lda
    :param y: the label
    :return:  None
    '''
    import matplotlib.pyplot as plt
    colors = 'rgb'
    markers = 'o*s'
    for target, color, marker in zip([1, 2, 3], colors, markers):
        pos = (y == target).ravel()
        X = converted_X[pos, :]
        plt.scatter(X[:, 0], X[:, 1], color=color, marker=marker,
                   label="Label %d" % target)
    plt.legend(loc="best")
    plt.suptitle(" After LDA")
    plt.show()


import numpy as np

x_data = np.loadtxt("ldaData.txt")
y_data = np.zeros(137)
print(x_data.shape, y_data.shape)
for i in range(50):
    y_data[i]=1
for i in range(50,93):
    y_data[i]=2
for i in range(93,137):
    y_data[i]=3
lda = discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(x_data, y_data)
pre = lda.predict(x_data)
num = 0
for i in range(137):
    if y_data[i] != pre[i]:
        num = num+1
print(num)
converted_X = np.dot(x_data, np.transpose(lda.coef_)) + lda.intercept_
plot_LDA(converted_X, y_data)
plot_LDA(x_data, y_data)