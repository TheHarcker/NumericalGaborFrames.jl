include("../utils.jl")
include("../modules/ghosh_selvan.jl")
include("../modules/zibulski_zeevi.jl")
include("../modules/lyubarskii_nes.jl")

import .GhoshSelvan as gs
import .ZibulskiZeevi as zz

compute = false

print_progress = 1
B_symmetric = true

T1 = Float64
T2 = Double64

optim_iter_M = 20
optim_iter = 20

beta_max = 10
dbeta = 0.01

pmax = 5
qmax = 5
fractions = generate_reduced_fractions_below_one(pmax, qmax)

dw = 0.001
dt = 0.01
dv = 0.01

for m in 2:5
    if compute
        phi_hat(x) = bspline_hat(m, x)
        compact_support_hat = (-200, 200)
        compute_sums(beta, w) = gs.compute_sums_generic(phi_hat, compact_support_hat, beta, w)

        beta_gs = uniform_spaced_values(dbeta, beta_max, dbeta; T=T1)
        alpha_gs = gs.frame_set_max_alpha(compute_sums, beta_gs, dw; B_symmetric, optim_iter_M, print_progress, T2)
        alpha_gs .*= beta_gs

        save_alpha_beta("data/frame_sets/abc_gs_bspline_$m.npz", alpha_gs, beta_gs)

        phi = x -> bspline(m, x)
        compact_support = (-m/2, m/2)
        A_zz, B_zz, alpha_zz, beta_zz = zz.frame_bounds(phi, compact_support, fractions, T1[], beta_gs, dt, dv; beta_max, optim_iter, print_progress, T2)
        alpha_zz = alpha_zz .* beta_zz

        save_bounds("data/frame_bounds/abc_zz_bspline_$m.npz", A_zz, B_zz, alpha_zz, beta_zz)
    else
        alpha_gs, beta_gs = load_alpha_beta("data/frame_sets/abc_gs_bspline_$m.npz")
        A_zz, B_zz, alpha_zz, beta_zz = load_bounds("data/frame_bounds/abc_zz_bspline_$m.npz",)
    end

    is_frame_zz = is_gabor_frame(A_zz, B_zz, min_tol=1e-20)

    plt = plot(beta_gs, alpha_gs, label = "GS max alpha")
    scatter!(beta_zz[.!is_frame_zz], alpha_zz[.!is_frame_zz], markersize=1, markerstrokewidth=0, label = "ZZ is not frame")
    scatter!(beta_zz[is_frame_zz], alpha_zz[is_frame_zz], markersize=1, markerstrokewidth=0, label = "ZZ is frame")
    xlabel!("\\beta")
    ylabel!("\\alpha \\beta")
    display(plt)
    savefig(plt, "plots/abc_gs_zz_bspline_$m.png")
end
