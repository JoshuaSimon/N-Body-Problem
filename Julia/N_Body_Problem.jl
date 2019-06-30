# N_Body_Problem.jl
# include(raw"d:\Dokumente\VS Code\Julia\N-Body-Problem\N_Body_Problem.jl")

using LinearAlgebra

# Number of Bodys
n = 3

# Time
delta_t = 0.1
t_max = 100
t = Int(div(t_max, delta_t))

# Masses
m = [0.9999, 0.00005, 0.00005]
M = sum(m)

# Gravity
G = 2.0

# Container for position R, velocity V and acceleration A
R = Array{Float64}(undef, (n, t, 2))
V = Array{Float64}(undef, (n, t, 2))
A = Array{Float64}(undef, (n, t, 2))

# Initial values for position and velocity
R[1,1,:] = [0, 0]
R[2,1,:] = [1, 0]
R[3,1,:] = [1.2, 0]

V[1,1,:] = [0.1, 0]
V[2,1,:] = [0, 1.4]
V[3,1,:] = [0, 0.8]

#= Alternativ: 
R[:,1,:] = [0 0; 1 0; 1.2 0] 
=#

function acceleration(R, G, m, i, t)
    a_new = []

    for j = 1:length(R[:,1,1])
        if i == j
            continue
        end

        r_i = R[i, t, :]
        r_j = R[j, t, :]
        r_ij = r_j - r_i
        
        if norm(r_ij) == 0
            println(i, "   ", j, "   ", r_i, r_j)
        end

        #println(r_ij)

        a_i = r_ij * G * m[j] * 1/norm(r_ij)^3

        push!(a_new, a_i)
    end

    return sum(a_new)

end

function euler_method(delta_t, V, R, A)
    v_new = V[:,t,:] + A[:,t,:] * delta_t
    r_new = R[:,t,:] + V[:,t,:] * delta_t

    return v_new, r_new
end



for time = 1:10

    #println(hcat(map(i -> acceleration(R, G, m, i, time), 1:n)...))
    
    A[:,t,:] = hcat(map(i -> acceleration(R, G, m, i, time), 1:n)...)'
    println(A[:,t,:])
    println()
    
    println(euler_method(delta_t, V, R, A))
    v_new, r_new = euler_method(delta_t, V, R, A)
    println("Wheeler done!")
    println()

    V[:,time+1,:] = v_new
    R[:,time+1,:] = r_new

    if time == t 
        break
    end

end


println("Fuck")
