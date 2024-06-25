include("../utils.jl")
include("../modules/ghosh_selvan.jl")

import .GhoshSelvan as gs

compute = false

T1 = Float64
T2 = Double64
optim_iter_M = 20
print_progress = 1

dbeta = 0.01
dw = 0.001
compact_support_hat = (-200, 200)

beta_min = 95
beta_max = 100

function find_alpha_peaks_non_integer(
    alpha::Vector{<:Real}, 
    beta::Vector{<:Real};
    tol::Real = 1e-3
)::Vector{Int}
    peaks_idx = Vector{Int}()

    beta_min_int = floor(Int, minimum(beta))
    beta_max_int = ceil(Int, maximum(beta))

    for n in beta_min_int:(beta_max_int - 1)
        bools = n + tol .< beta .< n + 1 - tol
        push!(peaks_idx, argmax(alpha .* bools))
    end

    return peaks_idx
end

for m in 2:5
    phi_hat = x -> bspline_hat(m, x)
    compute_sums(beta, w) = gs.compute_sums_generic(phi_hat, compact_support_hat, beta, w)

    file_path = "data/frame_sets/abc_gs_high_beta_bspline_$m.npz"
    if compute
        beta = uniform_spaced_values(beta_min, beta_max, dbeta; T=T1)
        alpha = gs.frame_set_max_alpha(compute_sums, beta, dw; optim_iter_M, print_progress, T2)
        alpha .*= beta

        save_alpha_beta(file_path, alpha, beta)
    else
        alpha, beta = load_alpha_beta(file_path)
    end

    peaks_idx = find_alpha_peaks_non_integer(alpha, beta)
    peaks_alpha = alpha[peaks_idx]
    peaks_beta = beta[peaks_idx]

    println(collect(zip(peaks_beta, peaks_alpha)))

    plt = plot(beta, alpha, label = "GS max alpha")
    scatter!(peaks_beta, peaks_alpha, label = "Peaks")
    xlabel!("\\beta")
    ylabel!("\\alpha \\beta")
    display(plt)
    savefig(plt, "plots/abc_gs_high_beta_bspline_$m.png")
end