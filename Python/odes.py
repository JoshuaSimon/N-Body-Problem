import numpy as np

from collections.abc import Iterable
from collections import deque
from functools import wraps
from typing import NamedTuple, Callable, Dict, Any, Tuple
from math import ceil

from itertools import count


class IvpSolution(NamedTuple):
    length: int
    ts: np.array
    ys: np.array


class Ivp(NamedTuple):
    f: Callable[[float, np.array], np.array]
    t_0: float
    y_0: np.array
    problem_params: Dict[str, Any]
    original_y_shape: Tuple[int]


def ivp(t_0: float, y_0: np.array, **problem_params):
    s = y_0.shape

    def unflatten_y(f):
        @wraps(f)
        def f_unflattened(t_0: float, y_0: np.array, **kwargs):
            return f(t_0, y_0.reshape(s), **kwargs)
        return f_unflattened

    return lambda f: Ivp(unflatten_y(f), t_0, y_0.flatten(), problem_params, s)


def solve_ivp(solver: Callable[[Ivp, Any], IvpSolution], **solver_params):
    def solve(ivp: Ivp):
        @wraps(ivp.f)
        def sol():
            solution = solver(ivp, **solver_params)
            # print((len(solution.ts), *ivp.original_y_shape))
            actual_ys = np.array(solution.ys).reshape((
                len(solution.ts), *ivp.original_y_shape))
            return IvpSolution(solution.length, solution.ts, actual_ys)
        return sol
    return solve


def eulers_method(ivp: Ivp, t_end, step_size):
    t_0 = ivp.t_0
    steps = ceil(((t_end - t_0) / step_size))
    ts_ys = np.zeros((steps + 1, ivp.y_0.size + 1))
    ts_ys[0, :] = np.array((ivp.t_0, *ivp.y_0))
    f = ivp.f
    p = ivp.problem_params
    for k in range(steps):
        y_k = ts_ys[k, 1:]
        t_k1 = t_0 + (k+1) * step_size
        y_k1 = y_k + step_size * f(t_k1, y_k, **p)
        ts_ys[k+1, 0] = t_k1
        ts_ys[k+1, 1:] = y_k1
    ts = ts_ys[:, 0]
    ys = ts_ys[:, 1:]
    return IvpSolution(len(ts), ts, ys)


def explicit_rk4(ivp: Ivp, t_end, step_size):
    """ Explicit classic 4-th order runge-kutta method """
    t_0 = ivp.t_0
    steps = ceil((t_end - t_0) / step_size)
    # first column is time, remainder is y
    ts_ys = np.zeros((steps + 1, ivp.y_0.size + 1))
    ts_ys[0, :] = np.array((ivp.t_0, *ivp.y_0))
    f = ivp.f
    p = ivp.problem_params
    for k in range(steps):
        t_k1 = t_0 + (k + 1) * step_size
        y_k = ts_ys[k, 1:]
        k_1 = f(t_k1, y_k, **p)
        k_2 = f(t_k1 + step_size / 2, y_k + step_size / 2 * k_1, **p)
        k_3 = f(t_k1 + step_size / 2, y_k + step_size / 2 * k_2, **p)
        k_4 = f(t_k1 + step_size, y_k + step_size * k_3, **p)
        y_k1 = y_k + step_size * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
        ts_ys[k+1, 0] = t_k1
        ts_ys[k+1, 1:] = y_k1

    ts = ts_ys[:, 0]
    ys = ts_ys[:, 1:]
    return IvpSolution(len(ts), ts, ys)


def dopri54(ivp: Ivp, t_end, step_size_0, eps_target):
    """ DOPRI 5(4)
    Dormand-Prince method:
    Embedded ODE solver based on 4th/5th order explicit rk4s. Tries to control
    the step size in such a way that the error is absolutely bounded by eps_target.
    """
    t_0 = np.array(ivp.t_0)
    # first column is time, remainder is y
    ts = deque([t_0])
    ys = deque([ivp.y_0])
    f = ivp.f
    p = ivp.problem_params
    step_size = step_size_0
    alpha = 0.9
    order = 5
    t_k = t_0
    exit_condition = False
    for k in count(1):
        t_k1 = t_k + step_size
        y_k = ys[-1]
        k_1 = f(t_k1, y_k, **p)
        k_2 = f(t_k1 + 1/5 * step_size, y_k + step_size * 1/5 * k_1, **p)
        k_3 = f(t_k1 + 3/10 * step_size, y_k +
                step_size * (3/40 * k_1 + 9/40 * k_2), **p)
        k_4 = f(t_k1 + 4/5 * step_size, y_k + step_size *
                (44/45 * k_1 - 56/15 * k_2 + 32/9 * k_3), **p)
        k_5 = f(t_k1 + 8/9 * step_size, y_k + step_size * (
            19372/6561 * k_1 - 25360/2187 * k_2 + 64448/6561 * k_3 - 212/729 * k_4), **p)
        k_6 = f(t_k1 + step_size, y_k + step_size * (
            9017/3168 * k_1 - 355/33 * k_2 + 46732/5247 * k_3 + 49/176 * k_4 - 5103/18656 * k_5), **p)
        k_7 = f(t_k1 + step_size, y_k + step_size * (
            35/384 * k_1 + 500/1113 * k_3 + 125/192 * k_4 - 2187/6784 * k_5 + 11/84 * k_6), **p)
        y_k1_5 = y_k + step_size * (
            35/384 * k_1 + 500/1113 * k_3 + 125 /
            192 * k_4 - 2187/6784 * k_5 + 11/84 * k_6)
        y_k1_4 = y_k + step_size * (
            5179/57600 * k_1 + 7571/16695 * k_3 +
            393/640 * k_4 - 92097/339200 * k_5 + 187/2100 * k_6 + 1/40 * k_7)
        error_estimate = np.linalg.norm(y_k1_5 - y_k1_4)
        if error_estimate != 0:
            step_size = max(min(5*step_size, alpha * step_size *
                                (eps_target / abs(error_estimate))**(1/order)), 0.001)
        ts.append(np.array(t_k1))
        ys.append(y_k1_5)
        t_k = t_k1
        if exit_condition:
            break
        if t_k1 + step_size >= t_end:
            step_size = t_end - t_k1
            exit_condition = True
        if step_size <= 0:
            raise ValueError("Internal error: negative step size")

    n = len(ts)
    ts = np.vstack(ts).reshape((n,))
    ys = np.vstack(np.array(ys))
    return IvpSolution(n, ts, ys)
