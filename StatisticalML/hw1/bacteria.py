import numpy as np
import matplotlib.pyplot as plt
def markov(iterator):
    init_array = np.random.rand(1,2)
    nor_array = init_array/np.sum(init_array)
    transfer_matrix = np.array([[0.45, 0.55],
                               [0.023, 0.977]])

    temp = nor_array
    for i in range(iterator):
        res = np.dot(temp, transfer_matrix)
        print (i, "\t", res)
        temp = res
def KL(p,q):
    print( np.sum(p*np.log2(p/q), axis = 1) )
def Entrop(p):
    print( np.sum(p * np.log2(1 / p), axis = 1) )

p=np.array([[2/9, 1/18, 1/3, 1/9, 1/18, 2/9],
            [5/18, 1/3, 1/18, 1/18, 2/9, 1/18],
            [1/6, 1/6, 2/9, 1/9, 1/6, 1/6]])

#q = 1/6
#KL(p,q)
p = np.array([[0.02, 0.67, 0.23, 0.08]])
e = np.array([[0.25, 0.25, 0.25, 0.25]])
Entrop(e)
