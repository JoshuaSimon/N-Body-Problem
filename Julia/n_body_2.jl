using LinearAlgebra
using Test

# Initial system state

# Time settings
Δt = 0.1
t_max = 250
t = Int(div(t_max, Δt))

# Masses of bodies
m = [0.999, 0.0005, 0.0005]
M = sum(m)
n = length(m)

# Gravitational constant
G = 1.0

# Matrices of shape [body, time, value]
R = zeros(n, t, 2) # Array{Float64}(undef, 
V = zeros(n, t, 2)
A = zeros(n, t, 2)

# Initial values for position and velocity
R[:, 1, :] = [0 0; 1 0; 2 0] 
V[:, 1, :] = [0.25 0; 0 1; 0 0.7]


""" Compute acceleration vector for timestep `t` and body `i`

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


"""Compute complete acceleration matrix for timestep `t`

See `acceleration_i` for details on params

# Returns
``A(t)`` Acceleration matrix of timestep `t`

"""
function acceleration(R, G, m, t)
    return hcat(map(i -> acceleration_i(R, G, m, t, i), 1:n)...)'
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

#= 
for time = 2:10
    A[:, t, :] = hcat(map(i -> acceleration_i(R, G, m, i, time), 1:n)...)'
end =#
