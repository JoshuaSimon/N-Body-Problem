# 2_Body_Problem_V3.py

from math import sqrt
import matplotlib.pyplot as plt
import operator


class Vector():
    """ n-dimensional Vector """
    def __init__(self, *components):
        self._components = components

    def __str__(self):
        return str(self._components)

    __repr__ = __str__

    def two_vector_elementwise(self, other, func):
        if len(self) != len(other):
            raise ValueError("Dimensions of vectors are different")
        return Vector(*[func(s, o) for (s, o) in zip(self._components, other._components)])

    def elementwise(self, func):
        return Vector(*[func(x) in self._components])

    def __sub__(self, other):
        return self.two_vector_elementwise(other, operator.sub)

    def __add__(self, other):
        return self.two_vector_elementwise(other, operator.add)

    @property
    def norm(self):
        sqrt(sum(x**2 for x in self._components))

    __abs__ = norm

    def __getitem__(self, index):
        return self._components[index]

    def __setitem__(self, index, value):
        self._components[index] = value



def euler (delta_t, i, v_i, R, m, G):
    """ Euler method to solve ODEs """
    def new_r(component):
        return R[i][-1][component] + v_i[-1][component] * delta_t

    def new_v(component): 
        return v_i[-1][component] + a[component] * delta_t

    a = a_nd(R, G, m)
    v_i_new = [new_v(component) for component in range(2)]
    r_new = [new_r(component) for component in range(2)]
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
    a_x_new = []
    a_y_new = []
    for i in range(len(R)):
        for j in range(len(R)):
            if i == j: continue
            r_ij = [r_j - r_i for (r_i, r_j) in (R[i][-1], R[j][-1])]

            r = Vector(r_ij[0], r_ij[1])

            a_ix = G * m[j] * r_ij[0] / vector_abs(r)
            a_iy = G * m[j] * r_ij[1] / vector_abs(r)
            a_x_new.append(a_ix)
            a_y_new.append(a_iy)

            #a_i = r.elementwise(lambda x_n: G * m[j] * x_n / r.norm)
            #a_i = r_ij.elementwise(lambda x_n: G * m[j] * x_n / r_ij.norm)
            #a_new.append(a_i)
    print(sum(a_x_new))
    print(sum(a_y_new))
    print(a_x_new)
    print(a_y_new)
    a = [sum(a_x_new), sum(a_y_new)]
    return a


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