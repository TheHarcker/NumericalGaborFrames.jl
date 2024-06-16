import Pkg
Pkg.add("Plots")
Pkg.add("LinearAlgebra")
Pkg.add("Plots")
Pkg.add("DoubleFloats")
Pkg.add("Optim")
Pkg.add("NPZ")
Pkg.add("CSV")

include("../utils.jl")
include("../modules/ghosh_selvan.jl")
import .GhoshSelvan as gs
using DelimitedFiles

T1 = Float64
T2 = Double64
optim_iter_M = 10
print_progess = 0
    
println("ARGS:", ARGS)
m_min::Int = parse(Int64, ARGS[1])
m_max::Int = parse(Int64, ARGS[2])
beta_min::Int = parse(Int64, ARGS[3])
beta_max::Int = parse(Int64, ARGS[4])
m_range = m_min:10:m_max
beta_range = beta_min:beta_max
dbeta = 1e-2
dw = 1e-3

table = zeros(length(m_range), length(beta_range))

Threads.@threads for i in 1:length(m_range)
    m = m_range[i]
    tol = 1/(2 * m)
    d = Base.max(10^(15/m) / pi, 10)
    compact_support = (-d, d)
    phi_hat(x) = bspline_hat(m, x)
    compute_sums(beta, w) = gs.compute_sums_generic(phi_hat, compact_support, beta, w)
    
    Threads.@threads for j in 1:length(beta_range)
        local beta_int = beta_range[j]
        
        beta_values = uniform_spaced_values(beta_int + tol, beta_int + 1 - tol, dbeta; T=T1)
        alpha_beta_values = beta_values .* gs.frame_set_max_alpha(compute_sums, beta_values, dw; optim_iter_M, print_progess, T2)        
        peak = maximum(alpha_beta_values)
        table[i,j] = peak

        println("finished m: ", m, " beta: ", beta_int)
    end
end

writedlm("../Data/tables/table_m_$m_min-$(m_max)_beta_$beta_min-$beta_max.csv",  table, ',')
print(table)
