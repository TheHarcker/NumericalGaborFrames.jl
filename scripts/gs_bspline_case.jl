include("../utils.jl")
include("../modules/ghosh_selvan.jl")

using .GhoshSelvan

#setprecision(BigFloat, 128)
T1 = Float64
T2 = Double64
print_progress = 1
B_symmetric = true
optim_iter_M = 100
optim_iter_s2 = 100

m = 2
compute_sums = (beta, w) -> compute_sums_bspline(m, beta, w)

alpha_max = 5
beta_max = 10
dalpha = 0.005
dbeta = 0.01
dw = 0.001

compute = true
if compute

    A, B, alpha, beta = frame_bounds_grid(compute_sums, dalpha, dbeta, dw, alpha_max, beta_max; optim_iter_M, optim_iter_s2, B_symmetric, print_progress, T1, T2)
    save_bounds("../Data/frame_bounds/gs_bspline_case_$m.npz", A, B, alpha, beta)
else
    A, B, alpha, beta = load_bounds("../Data/frame_bounds/gs_bspline_case_$m.npz")
end

is_frame = is_gabor_frame(A, B)
plt = scatter(alpha[is_frame], beta[is_frame], markersize=1, markerstrokewidth=0, xlabel="\\alpha", ylabel="\\beta")
display(plt)
savefig(plt, "../Figures/julia/frame_set_bpsline_case_$m.png")
