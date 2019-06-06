# 2_Body_Problem.py

from math import sqrt
import matplotlib.pyplot as plt

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
    """ acceleration function for two bodys"""

    # Distance vector r with components rx, ry, rz
    rx = r2[0] - r1[0]
    ry = r2[1] - r1[1]
    #rz = r2[2] - r1[2]
    rz = 0

    a1 = -1 * g * M /sqrt(rx**2 + ry**2 + rz**2)**3 * rx        # Body N = 1
    a2 = -1 * g * M /sqrt(rx**2 + ry**2 + rz**2)**3 * ry        # Body N = 2
    return a1, a2

def euler (t_n, v1, r1, r2):
    """ Euler method to solve ODEs """
    
    ax, ay = a(g, M, r1, r2)
    v1_x_new = v1[0] + ax * delta_t
    r1_x_new = r1[0] + v1[0] * delta_t

    v1_y_new = v1[1] + ay * delta_t
    r1_y_new = r1[1] + v1[1] * delta_t
    return v1_x_new, r1_x_new, v1_y_new, r1_y_new


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
r1_start = [1, 0, 0]
v1_start = [0, 0, 0]
r2_start = [0, 0, 0]
v2_start = [0, -1, 0] 

# Gravity
g = 1.0


# 2 Calculation
# -------------
v1 = v1_start
v2 = v2_start
r1 = r1_start
r2 = r2_start

# Loop over time steps (start at 0, end at t_max, step = delta_t)
for i in range(2):
    v1_x_new, r1_x_new, v1_y_new, r1_y_new = euler(i, v1, r1, r2)       # Body N = 1
    v2_x_new, r2_x_new, v2_y_new, r2_y_new = euler(i, v2, r1, r2)       # Body N = 2

    v1 = [v1_x_new, v1_y_new]       # New velocity vector for Body N = 1
    v2 = [v2_x_new, v2_y_new]       # New velocity vector for Body N = 2
    r1 = [r1_x_new, r1_y_new]       # New position vector for Body N = 1
    r2 = [r2_x_new, r2_y_new]       # New position vector for Body N = 2

    print("Time = ", i)
    print(v1)
    print(v2)
    print(r1)
    print(r2)



