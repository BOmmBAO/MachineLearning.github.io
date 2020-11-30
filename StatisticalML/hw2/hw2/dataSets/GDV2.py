import numpy as np
def main():
    list1 = np.array([[1, 1, 1],
                     [0, 4, 0],
                     [0, 0, -8]])
    list2 = np.array([[1, 1, 1],
                      [0,4,0],
                      [0,0,8]])
    print (np.linalg.det(list1)/np.linalg.det(list2))
main()