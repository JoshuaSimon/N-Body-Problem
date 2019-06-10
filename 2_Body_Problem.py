# 2_Body_Problem.py

from math import sqrt
import matplotlib.pyplot as plt
import operator


class Vector():
    """ n-dimensional Vector """
    def __init__(self, *components):
        self._components = list(components)

    def __str__(self):
        return f"Vector{self._components}"

    __repr__ = __str__

    def two_vector_elementwise(self, other, func):
        if len(self) != len(other):
            raise ValueError("Dimensions of vectors are different")
        return Vector(*[func(s, o) for (s, o) in zip(self._components, other._components)])

    def elementwise(self, func):
        return Vector(*[func(x) for x in self._components])

    def __sub__(self, other):
        return self.two_vector_elementwise(other, operator.sub)

    def __add__(self, other):
        return self.two_vector_elementwise(other, operator.add)

    @property
    def norm(self):
        return sqrt(sum(x**2 for x in self._components))

    __abs__ = norm

    def __getitem__(self, index):
        return self._components[index]

    def __setitem__(self, index, value):
        self._components[index] = value

    def __len__(self):
        return len(self._components)

    dim = __len__
# just a template atm and not working
def rk4 (a, t_n, v_n, r_n, delta_t):
    """ Fourth-order Runge-Kutta method (RK4) """

    a_n = a(t_n)
    v1_tilde = a_n * delta_t                # a_n = a(t_n) = acceleration at the moment 
    r1_tilde = v_n * delta_t
    v2_tilde = a(t_n + 1/2 * delta_t, r_n + 1/2 * r1_tilde) * delta_t
    r2_tilde = (v_n + 1/2 * v1_tilde) * delta_t
    v3_tilde = a(t_n + 1/2 * delta_t, r_n + 1/2 * r2_tilde) * delta_t
    r3_tilde = (v_n + 1/2 * v2_tilde) * delta_t
    v4_tilde = a(t_n + delta_t, r_n + r3_tilde) * delta_t
    r4_tilde = (v_n + 1/2 * v3_tilde) * delta_t

    v_n1 = v_n + 1/6 * (v1_tilde + 2*v2_tilde + 2*v3_tilde + v4_tilde)
    r_n1 = r_n + 1/6 * (r1_tilde + 2*r2_tilde + 2*r3_tilde + r4_tilde)

    return v_n1, r_n1
    

def a (g, M, r1, r2):
    """ acceleration function """

    # Distance vector r with components rx, ry, rz
    rx = r2[0] - r1[0]
    ry = r2[1] - r1[1]
    #rz = r2[2] - r1[2]
    rz = 0

    ax = -1 * g * M /sqrt(rx**2 + ry**2 + rz**2)**3 * rx
    ay = -1 * g * M /sqrt(rx**2 + ry**2 + rz**2)**3 * ry
    return ax, ay


def euler (delta_t, i, v_i, R, m, G):
    """ Euler method to solve ODEs """
    def new_r(component):
        return R[i][-1][component] + v_i[-1][component] * delta_t

    def new_v(component): 
        return v_i[-1][component] + a[component] * delta_t

    a = a_nd(R, G, m)
    v_i_new = [new_v(component) for component in range(len(v_i[0]))]
    r_new = [new_r(component) for component in range(len(R[0][0]))]
    return v_i_new, r_new


def vector_abs(v):
    return sqrt(sum(x**2 for x in v))


def a_nd(R, G, m):
    """ Acceleration of next timestep for 1 body in a system of n bodies
    Acceleration as x and y components
    Params:
        R: Vector of vector of position tuples of elements
        G: Gravitational constant
        m: Vector of masses
    """
    a_new = []
    for i in range(len(R)):
        for j in range(len(R)):
            if i == j: continue
            r_ij = [r_j - r_i for (r_i, r_j) in (R[i][-1], R[j][-1])]
            a_i = r_ij.elementwise(lambda x_n: G * m[j] * x_n / r_ij.norm)
            a_new.append(a_i)
    return sum(a_new)


# 1 Input Data
# ---------------
# Number of bodys
n = 2

# Maximum integration time
t_max = 2.0

# Time step length
delta_t = 0.1

# Mass
m = [1.0, 1.0]              # [Mass of Body1, Mass of Body2]
M = sum(m, 0)
my = m[0]*m[1] / M

# Initial position r and velocity v of the two bodys 
r1_start = Vector(1, 0)
v1_start = Vector(0, 0)
r2_start = Vector(0, 0)
v2_start = Vector(0, -1) 

r_start = [[r1_start], [r2_start]]
v_start = [[v1_start], [v2_start]]

# Gravity
G = 1.0


# 2 Calculation
# -------------
R = r_start
V = v_start

# Loop over time steps (start at 0, end at t_max, step = delta_t)
for t in range(0, int(t_max//delta_t)):
    print()
    for i in range(n):
        r_i_new, v_i_new = euler(delta_t, i, V[i], R, m, G)
        
        R[i].append(r_i_new)
        V[i].append(v_i_new)


for n in range(n):
    plt.plot(*list(zip(*R[n])), "o", label=r"$R_{}$".format(n))

plt.legend()
plt.show()
