using LinearAlgebra
using Test
using Plots

# Initial system state

# Time settings
Δt = 0.1
t_max = 100
t = Int(div(t_max, Δt))

# Masses of bodies
m = [0.999, 0.0005, 0.0005]
M = sum(m)
n = length(m)

# Gravitational constant
G = 1.0

# Matrices of shape [body, time, value]
R = zeros(n, t, 2)
V = zeros(n, t, 2)
A = zeros(n, t, 2)

# Initial values for position and velocity
R[:, 1, :] = [0 0; 1 0; 2 0] 
V[:, 1, :] = [0.25 0; 0 1; 0 0.7]


"""
 Compute acceleration vector for timestep `t` and body `i`

# Arguments
- `R`: Matrix of shape `[body, time, position vector]`
- `G`: Gravitational constant
- `m`: Vector of masses of bodies
- `t`: Index of time step into `R`
- `i`: Index of body into `R`/`G`/`m`

# Returns
``a_i(t)`` Acceleration vector of body `i` and timestep `t`
"""
function acceleration_i(R, G, m, t, i)
    a_new = []

    for j = 1:length(R[:,1,1])
        if i == j
            continue
        end
        r_i = R[i, t, :]
        r_j = R[j, t, :]
        r_ij = r_j - r_i

        a_ij = r_ij * G * m[j] / norm(r_ij)^3
        push!(a_new, a_ij)
    end
    return sum(a_new)
end


"""
Compute complete acceleration matrix for timestep `t`

See `acceleration_i` for details on params

# Returns
``A(t)`` Acceleration matrix of timestep `t`
"""
function acceleration(R, G, m, t)
    return hcat(map(i->acceleration_i(R, G, m, t, i), 1:n)...)'
end

@testset "acceleration" begin
    time = 2
    m = [0.999, 0.0005, 0.0005]
    n = 3
    R = zeros(n, time, 2)
    R[:, 1, :] = [0 0; 1 0; 2 0]
    G = 1.0
    t = 1
    i = 1
    @test acceleration_i(R, G, m, t, i) == [0.000625, 0.0]
    i = 2
    @test acceleration_i(R, G, m, t, i) == [-0.9985, 0.0]
    i = 3
    @test acceleration_i(R, G, m, t, i) == [-0.25025, 0.0]

    @test acceleration(R, G, m, t) == [
        0.000625 0.0; 
        -0.9985 0.0;
        -0.25025 0.0]
end


"""
Euler-Method: Compute the new position and 
velocity for all the bodys in the system for the
next timestep.

# Arguments
- `Δt`: timestep length
- `t`: Index of time step into `V`, `R` and `A`
- `V`: Matrix of shape `[body, time, velocity vector]`
- `R`: Matrix of shape `[body, time, position vector]`
- `A`: Matrix of shape `[body, time, acceleration vector]`

# Returns
``v_new(t), r_new(t)`` Velocitiy and position of one body
at the next timestep
"""
function euler_method(Δt, t, V, R, A)
    v_new = V[:,t,:] + A[:,t,:] * Δt
    r_new = R[:,t,:] + V[:,t,:] * Δt

    return v_new, r_new
end


"""
Euler-Cormer- Method: Compute the new position and 
velocity for all the bodys in the system for the
next timestep.

# Arguments
- `Δt`: timestep length
- `t`: Index of time step into `V`, `R` and `A`
- `V`: Matrix of shape `[body, time, velocity vector]`
- `R`: Matrix of shape `[body, time, position vector]`
- `A`: Matrix of shape `[body, time, acceleration vector]`

# Returns
``v_new(t), r_new(t)`` Velocitiy and position of one body
at the next timestep
"""
function euler_cormer(Δt, t, V, R, A)
    v_new = V[:,t,:] + A[:,t,:] * Δt
    r_new = R[:,t,:] + 0.5 * (V[:,t,:] + v_new) * Δt

    return v_new, r_new
end


"""
Verlet-Algotihm: Compute the new position and 
velocity for all the bodys in the system for the
next timestep. Speical case for t = t_start.
(Here t_start = 1).

# Arguments
- `Δt`: timestep length
- `t`: Index of time step into `V`, `R` and `A`
- `V`: Matrix of shape `[body, time, velocity vector]`
- `R`: Matrix of shape `[body, time, position vector]`
- `A`: Matrix of shape `[body, time, acceleration vector]`

# Returns
``v_new(t), r_new(t)`` Velocitiy and position of one body
at the next timestep
"""
function verlet(Δt, t, V, R, A)

    if t == 1
        r_help = R[:,1,:] - V[:,1,:] * Δt + 0.5 * A[:,1,:] * Δt^2
        r_new = 2 * R[:,t,:] - r_help + A[:,t,:] * Δt^2
        v_new = (r_new - r_help) / (2 * Δt)
    else
        r_new = 2 * R[:,t,:] - R[:,t-1,:] + A[:,t,:] * Δt^2
        v_new = (r_new - R[:,t-1,:]) / (2 * Δt)
    end

    return v_new, r_new
end

#= 
iterate over solvers; solve our example problem with each one
and animate each solution
=#

function animate_solver(solver, R, G, m, t, V, Δt)
    for time = 1:(t - 1)
        A[:, time, :] = acceleration(R, G, m, time)
        v_new, r_new = solver(Δt, time, V, R, A)
    
        V[:, time + 1, :] = v_new
        R[:, time + 1, :] = r_new
    end
    
    anim = @animate for t = 1:size(R, 2)
        Rx = [R[i, t, 1] for i in 1:size(R, 1)]
        Ry = [R[i, t, 2] for i in 1:size(R, 1)]
        scatter(Rx, Ry, xlims = (-5, 20), ylims = (-5, 5), label = ["Body $i" for i in 1:size(R, 1)])
    end every 2
    
    name = string(solver)
    gif(anim, "R_$name.gif", fps = 30)
end

for solver in [euler_cormer, euler_method, verlet]
    animate_solver(solver, R, G, m, t, V, Δt)
end


