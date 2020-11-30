from scipy.integrate import odeint
import numpy as np
from scipy.optimize import root,fsolve
def f3(x):
    return np.array([np.log(x[0]) + 1/np.log(2) + 2 * x[4],
                    np.log(x[1]) + 1/np.log(2) + 4 * x[4],
                    np.log(x[2]) + 1 / np.log(2) + 6 * x[4],
                    np.log(x[3]) + 1 / np.log(2) + 8 * x[4],
                    x[0]+2*x[1]+3*x[2]+4*x[3]])

sol3_root = root(f3,[0,0,0,0,0])
sol3_fsolve = fsolve(f3,[0,0,0,0,0])
print(sol3_fsolve)