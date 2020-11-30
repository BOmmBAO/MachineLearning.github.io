import numpy as np

x = np.zeros((20,), dtype=np.float32)
alpha = 0.00001
max_iter_count = 10000
it_num = 1
dif = 1


def rosenbrock_der(x):
    der = np.zeros((20,), dtype=np.float32)
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) + 2 * (x[0] - 1)
    der[19] = 200 * (x[-1] - (x[-2] ** 2))

    for i in range(1, 19):
        der[i] = 200 * (x[i] - x[i - 1] ** 2) - 400 * x[i] * (x[i + 1] - x[i] ** 2) + 2 * (x[i] - 1)

    return der


def rosenbrock(x):
    rosen = 0
    for i in range(19):
        rosen += 100.0 * (x[i + 1] - x[i] ** 2.0) ** 2.0 + (x[i] - 1) ** 2.0

    return rosen

while dif > 0.0001 and it_num < max_iter_count:
    rosen_old = rosenbrock(x)

    for i in range(20):
        x[i] = x[i] - alpha * rosenbrock_der(x)[i]

    rosen_new = rosenbrock(x)
    dif = rosen_old - rosen_new

    print("count: ", it_num, "the dif:", dif, "step:", alpha)
    it_num += 1

print(x)