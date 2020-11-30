import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt

def loadDataSet():
    data = np.loadtxt("iris-pca.txt")
    positive = np.array