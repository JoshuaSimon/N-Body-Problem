import numpy as np
import matplotlib.pyplot as plt
from odes import *
from numpy.random import default_rng

y_0 = np.array([1.0])


@solve_ivp(dopri54, t_end=1, step_size_0=0.1, eps_target=1e-6)
# @solve_ivp(explicit_rk4, t_end=1, step_size=0.4)
@ivp(t_0=0.0,
     y_0=y_0.flatten())
def exp_sys(t: np.array, y: np.array):
    return y


xs = np.linspace(0, 1)
ys = np.exp(xs)
solution = exp_sys()
print(solution.ts.shape, solution.ys.shape)
dt = np.diff(solution.ts)
plt.plot(np.array(range(len(dt))), dt)
plt.show()
for i, (t, y) in enumerate(zip(solution.ts, solution.ys)):
    plt.cla()
    plt.axis([0, 1, 0, 3])
    plt.plot(xs, ys)
    plt.plot(solution.ts[i], solution.ys[i], "ro")
    plt.plot(solution.ts[:i+1], solution.ys[:i+1], "r-")
    if i < len(dt):
        plt.pause(dt[i])
    else:
        plt.pause(0.1)
plt.show()
