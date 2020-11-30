x1 = [1, 2, 5 , 6, 8, 3,4,7,9,10]
x2 = [5,2,8,10,7,1,6,4,3,9]
y = [1,1,1,1,1,0,0,0,0,0]
#num = 10
import numpy as np
for i in range(1,11):
    num = 10
    for j in range(10):
        if ((x1[j] <= 2 and x2[j] <= i and y[j] == 1)or(x1[j] > 2 or x2[j] > i and y[j] == 0)):
            num = num-1
    a = 0.5 * np.log((10 - num) / num)
    print(num, a)
# for i in range(10):
#     if ((x1[i] <= 2 and x2[i] <= 4 and y[i] == 1)or(x1[i] > 2 or x2[i] > 4 and y[i] == 0)):
#         num = num-1
# a = 0.5*np.log((10-num)/num)
# print(num, a)
# # print(0.5*np.log(3/7))
# # print(1*(0.423**0.7))
import numpy as np
a = [3/5, 2/5]
H = 0.940+5/7*np.sum(a*np.log2(a))
print(H)