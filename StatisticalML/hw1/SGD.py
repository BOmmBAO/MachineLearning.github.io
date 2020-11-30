import numpy as np 

def cal_rosenbrock(x):

    #compute rosenbrock
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)



def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der


def armijo(x,error,alpha = 1):

    # armijo search alpha
    def loss(alpha,x,loss = 0):
        temp = []
        for i in range(20):
            temp.append(x[i])
        for i in range(20):
            temp[i] -= alpha * error[i]
        for i in range(19):
            loss += 100 * (temp[i + 1] - temp[i] ** 2) ** 2 + (temp[i] - 1) ** 2

        return loss

    def check(alpha,x):
        return loss(alpha,x) > loss_0 - sigma * alpha * np.dot(error,error)

    sigma = 0.02
    rho = 0.4
    loss_0 = loss(0,x)

    if check(alpha,x) == False:
        alpha = 1
    elif check(alpha,x):
        alpha = 0.02
        while check(alpha,x):
            alpha *= rho

    return alpha

def for_rosenbrock_func(max_iter_count = 10000):
    pre_x = np.zeros((20,),dtype=np.float32)
    loss = 19
    cnt = 0
    while loss > 0.001 and cnt < max_iter_count:
        error = np.zeros((20,), dtype=np.float32)

        #compute grade
        error = rosen_der(pre_x)

        #find best solution

        alpha = armijo(pre_x,error)

        for j in range(20):
            pre_x[j] -= alpha * error[j]

        loss = cal_rosenbrock(pre_x)  # 最小值为0

        print("count: ", cnt, "the loss:", loss,"step:",alpha)
        cnt += 1
    return pre_x

if __name__ == '__main__':
    w = for_rosenbrock_func()  
    print(w)