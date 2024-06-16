include("../utils.jl")
include("../modules/ghosh_selvan.jl")

import .GhoshSelvan as gs

compute = true
T1 = Float64
T2 = Float64
optim_iter_M = 20
optim_iter_s2 = 20
print_progress = 2

m = 5
B_symmetric = true

dalpha = 0.002
dbeta = 0.01
dw = 0.001

alpha_max = 5
beta_min = dbeta
beta_max = 10

# Compute and save or load old
file_path1 = "../Data/frame_sets/frame_set_gs_special_bspline_$m.npz"
file_path2 = "../Data/frame_sets/frame_set_gs_generic_bspline_$m.npz"
if compute
    compute_sums1(beta, w) = gs.compute_sums_bspline(m, beta, w)
    A1, B1, alpha1, beta1 = gs.frame_bounds_grid(compute_sums1, dalpha, dbeta, dw, alpha_max, beta_max; beta_min, optim_iter_M, optim_iter_s2, B_symmetric, print_progress, T1, T2)
    
    phi_hat(x) = bspline_hat(m, x)
    compact_support = (-200, 200)
    compute_sums2(beta, w) = gs.compute_sums_generic(phi_hat, compact_support, beta, w)
    A2, B2, alpha2, beta2 = gs.frame_bounds_grid(compute_sums2, dalpha, dbeta, dw, alpha_max, beta_max; beta_min, optim_iter_M, optim_iter_s2, B_symmetric, print_progress, T1, T2)

    save_bounds(file_path1, A1, B1, alpha1, beta1)
    save_bounds(file_path2, A2, B2, alpha2, beta2)
else
    A1, B1, alpha1, beta1 = load_bounds(file_path1)
    A2, B2, alpha2, beta2 = load_bounds(file_path2)
end

is_frame1 = is_gabor_frame(A1, B1)
is_frame2 = is_gabor_frame(A2, B2)

plt = scatter(alpha1[is_frame1], beta1[is_frame1], markersize=0.3, markerstrokewidth=0, legend=false)
xlabel!("\\alpha")
ylabel!("\\beta")
ylims!(0, beta_max)
display(plt)
savefig(plt, "../Figures/julia/frame_set_gs_special_bspline_$m.png")

plt = scatter(alpha2[is_frame2], beta2[is_frame2], markersize=0.3, markerstrokewidth=0, legend=false)
xlabel!("\\alpha")
ylabel!("\\beta")
ylims!(0, beta_max)
display(plt)
savefig(plt, "../Figures/julia/frame_set_gs_generic_bspline_$m.png")
