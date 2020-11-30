import numpy as np
import pandas as pd

class Bayes(object):
    def getTrainSet(self):
        dataSet = pd.read_csv('F://2020 SoSe//sml//ass2//hw2//dataSets//densEst1.txt')
        dataSetNP = np.array(dataSet)
        trainData = dataSetNP[:, 0:dataSetNP.shape[1] -1]
        labels = dataSetNP[:, dataSetNP.shape[1]-1]
        return trainData, labels
    def prior(self, labels ):
        labels = list(labels)
        P_y = {}
        for label in labels:
            P_y[label] = labels.count(label)/float(len(labels))
        return P_y

nb = Bayes()
trainData, labels = nb.getTrainSet()
result = nb.prior(labels)
print (result)