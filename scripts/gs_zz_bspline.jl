include("../utils.jl")
include("../modules/ghosh_selvan.jl")
include("../modules/zibulski_zeevi.jl")

import .GhoshSelvan as gs
import .ZibulskiZeevi as zz

compute = true

print_progress = 1
B_symmetric = true

T1 = Float64
T2 = Double64

optim_iter_M = 20
optim_iter_s2 = 20
optim_iter = 20

alpha_max = 5
beta_max = 5
pmax = 5
qmax = 5

dalpha = 0.01
dbeta = 0.01
dw = 0.001

dt = 0.01
dv = 0.01

for m in 2:5
    if compute
        phi_hat = x -> bspline_hat(m, x)
        compact_support_hat = (-200, 200)
        compute_sums(beta, w) = gs.compute_sums_generic(phi_hat, compact_support_hat, beta, w)
        A_gs, B_gs, alpha_gs, beta_gs = gs.frame_bounds_grid(compute_sums, dalpha, dbeta, dw, alpha_max, beta_max; optim_iter_M, optim_iter_s2, B_symmetric, print_progress, T1, T2)

        fractions = generate_reduced_fractions_below_one(pmax, qmax)
        phi = x -> bspline(m, x)
        compact_support = (-m/2, m/2)
        is_not_frame = nothing
        A_zz, B_zz, alpha_zz, beta_zz = zz.frame_bounds_grid(phi, compact_support, fractions, dalpha, dbeta, dt, dv, alpha_max, beta_max; optim_iter, is_not_frame, print_progress, T1, T2)

        save_bounds("../Data/frame_bounds/gs_bspline_$m.npz", A_gs, B_gs, alpha_gs, beta_gs)
        save_bounds("../Data/frame_bounds/zz_bspline_$m.npz", A_zz, B_zz, alpha_zz, beta_zz)
    else
        A_gs, B_gs, alpha_gs, beta_gs = load_bounds("../Data/frame_bounds/gs_bspline_$m.npz")
        A_zz, B_zz, alpha_zz, beta_zz = load_bounds("../Data/frame_bounds/zz_bspline_$m.npz")
    end

    is_frame_gs = is_gabor_frame(A_gs, B_gs)
    is_frame_zz = is_gabor_frame(A_zz, B_zz, min_tol=1e-20)

    plt = scatter(alpha_gs[is_frame_gs], beta_gs[is_frame_gs], markersize=1 , markerstrokewidth=0, label="GS is frame", alpha=0.5)
    scatter!(alpha_zz[.!is_frame_zz], beta_zz[.!is_frame_zz], markersize=1, markerstrokewidth=0, label="ZZ is not frame", alpha=0.5)
    scatter!(alpha_zz[is_frame_zz], beta_zz[is_frame_zz], markersize=1, markerstrokewidth=0, label="ZZ is frame", alpha=0.5)

    xlabel!("\\alpha")
    ylabel!("\\beta")
    display(plt)
    savefig(plt, "../Figures/julia/frame_set_bspline_$m.png")

    plt = scatter3d(alpha_gs[is_frame_gs], beta_gs[is_frame_gs], log10.(A_gs[is_frame_gs]), label="GS lower bond", markersize=1, markerstrokewidth=0, camera=(135,30), alpha=0.5)
    scatter3d!(alpha_gs[is_frame_gs], beta_gs[is_frame_gs], log10.(B_gs[is_frame_gs]), label="GS upper bound", markersize=1, markerstrokewidth=0, alpha=0.5)
    scatter3d!(alpha_zz[is_frame_zz], beta_zz[is_frame_zz], log10.(A_zz[is_frame_zz]), label="ZZ lower bound", markersize=1, markerstrokewidth=0, alpha=0.5)
    scatter3d!(alpha_zz[is_frame_zz], beta_zz[is_frame_zz], log10.(B_zz[is_frame_zz]), label="ZZ upper bound", markersize=1, markerstrokewidth=0, alpha=0.5)
    xlabel!("\\alpha")
    ylabel!("\\beta")
    display(plt)
    savefig(plt, "../Figures/julia/frame_bounds_bspline_$m.png")
end