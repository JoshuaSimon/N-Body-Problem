import mayavi
import numpy as np
import matplotlib.pyplot as plt
from odes import *
from numpy.random import default_rng
from mayavi import mlab
from numba import jit

gravitational_constant = 1.0
num_bodies = 20
radius = 10
angles = np.linspace(0, 2*np.pi, num=num_bodies+1)[:-1]
masses = np.matrix(np.ones(angles.shape)) / num_bodies
initial_positions = radius * np.array([np.cos(angles), np.sin(angles)]).T
initial_velocities = np.array([-np.sin(angles), np.cos(angles)]).T

plt.scatter(initial_positions[:, 0], initial_positions[:, 1])
plt.quiver(initial_positions[:, 0], initial_positions[:, 1],
           initial_velocities[:, 0], initial_velocities[:, 1])
plt.show()

y_0 = np.vstack([initial_positions, initial_velocities])
print(y_0)


# @solve_ivp(explicit_rk4, t_end=10, step_size=0.01)
@ solve_ivp(dopri54, t_end=100, step_size_0=0.05, eps_target=1e-7)
@ ivp(t_0=0.0,
      y_0=y_0,
      gravitational_constant=gravitational_constant,
      masses=masses,
      dimension=initial_positions[0].size)
def n_body_problem(t: np.array, y: np.array, gravitational_constant: float, masses: np.matrix, dimension: int):
    """ d-dimensional n body problem 
        Args:
            t: time
            y: matrix of size 2n × d such that the first n rows are the positions r
                and the second n rows are the velocities v for the n bodies in order
            m: vector with mass of each particle; e.g. m = np.matrix([1,2,3])
        Returns:
            dy/dt
        """
    d = dimension
    n = y.shape[0] // d
    # d = y.shape[1]
    r = y[:n, :]  # current positions
    v = y[n:, :]  # current velocities
    # calculate forces on each particle
    # find matrix with all possible products between masses
    m = masses.transpose() @ masses
    # print(f"{m=}")
    # find all distance differences; dimension of diff1 is n × n × d
    # print(f"{r=}")
    diff = np.array([r - r[k, :] for k in range(n)])
    # print(f"{diff=}")
    # remove zero rows / self interactions
    # diff = diff1[np.all(diff1 == 0, axis=2)].reshape((n, n-1, d))
    # calculate |r_i - r_j|³
    denom = np.sum(diff**2, axis=2) ** (3/2)
    # note that these are elementwise operations
    # denom will always contain zeros (on self interation)
    # we ignore the divide by zero
    with np.errstate(divide='ignore'):
        scalar_factor = gravitational_constant * m / denom
    # set the zero divisions to zero
    scalar_factor[np.isinf(scalar_factor)] = 0
    # actually calculate force vectors
    # print(f"{scalar_factor=}")
    # print(f"{diff[:, :, 0]=}")
    force = np.array([scalar_factor * diff[:, :, k] for k in range(d)])
    # print(f"{force=}")
    # find total force on each particle in this step; the dimension is d × n
    cum_force = np.sum(force, axis=2)
    # the acceleration of the particle is given by a = F/m
    acceleration = cum_force / masses
    return np.vstack([v, acceleration.transpose()]).flatten()


def get_body_color(i_body, n_bodies, cmap=plt.get_cmap('jet_r')):
    return cmap(i_body / n_bodies)


max_x = np.amax(initial_positions[:, 0])
max_y = np.amax(initial_positions[:, 1])
min_x = np.amin(initial_positions[:, 0])
min_y = np.amin(initial_positions[:, 1])

phi = np.linspace(0, 2*np.pi, num=10)
theta = np.linspace(0, np.pi, num=10)
P, T = np.meshgrid(phi, theta)
unit_sphere = (
    0.2 * np.cos(P)*np.sin(T),
    0.2 * np.sin(P)*np.sin(T),
    0.2 * np.cos(T))
meshes = []
for i in range(num_bodies):
    x = unit_sphere[0] + initial_positions[i, 0]
    y = unit_sphere[1] + initial_positions[i, 1]
    z = unit_sphere[2]  # + initial_positions[i, 2]
    c = get_body_color(i, num_bodies)[:3]
    meshes.append(mlab.mesh(x, y, z, color=c))


@mlab.animate
@jit
def anim():
    solution = n_body_problem()
    for i in range(1, len(solution.ts)):
        # plt.axis([min_x * 1.2, max_x * 1.2, min_y * 1.2, max_y * 1.2])
        positions_to_i = solution.ys[:i, :len(initial_positions), :]
        # print(positions_to_i)
        delta_t = solution.ts[i] - solution.ts[i-1]
        for j, mesh in enumerate(meshes):
            xs = positions_to_i[:, j, 0]
            ys = positions_to_i[:, j, 1]
            mesh.mlab_source.trait_set(
                x=unit_sphere[0] + xs[-1],
                y=unit_sphere[1] + ys[-1]
            )
            # c = get_body_color(j, num_bodies)
            # plt.plot(xs, ys, color=c)
            # plt.plot([xs[-1]], [ys[-1]], "o", color=c)
            yield


anim()
mlab.show()
"""
solution = n_body_problem()
plt.plot(np.array(range(0, len(solution.ts))), solution.ts)
plt.show()
for i in range(1, len(solution.ts)):
    plt.cla()
    plt.axis([min_x * 1.2, max_x * 1.2, min_y * 1.2, max_y * 1.2])
    positions_to_i = solution.ys[:i, :len(initial_positions), :]
    _, n_bodies, _ = positions_to_i.shape
    # print(positions_to_i)
    delta_t = solution.ts[i] - solution.ts[i-1]
    for j in range(n_bodies):
        xs = positions_to_i[:, j, 0]
        ys = positions_to_i[:, j, 1]
        c = get_body_color(j, n_bodies)
        plt.plot(xs, ys, color=c)
        plt.plot([xs[-1]], [ys[-1]], "o", color=c)
    plt.pause(delta_t/10)
plt.show()
"""
