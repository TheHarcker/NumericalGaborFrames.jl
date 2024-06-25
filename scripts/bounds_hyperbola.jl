# Run if script is executed directly
include("../utils.jl")
include("../modules/ghosh_selvan.jl")
include("../modules/zibulski_zeevi.jl")

import .ZibulskiZeevi as zz
import .GhoshSelvan as gs
using Plots

compute = false

m = 2

T1 = Float64
T2 = Double64
optim_iter_M = 20
optim_iter_s2 = 20
optim_iter = 20
print_progress = 2

B_symmetric = true

dalpha = 0.01
dbeta = 0.01
dw = 0.001

dt = 0.01
dv = 0.01

alpha_max = 5
beta_min = 0
beta_max = 10
fractions = [1//3]

gs_path = "data/frame_bounds/gs_hyperbola_bspline_$m.npz"
zz_path = "data/frame_bounds/zz_hyperbola_bspline_$m.npz"

if compute
    phi = x -> bspline(m, x)
    compact_support = (-m/2, m/2)
    is_not_frame = nothing
    A_zz, B_zz, alpha_zz, beta_zz = zz.frame_bounds_grid(phi, compact_support, fractions, dalpha, dbeta, dt, dv, alpha_max, beta_max; optim_iter, is_not_frame, print_progress, T1, T2)
    save_bounds(zz_path, A_zz, B_zz, alpha_zz, beta_zz)

    phi_hat(x) = bspline_hat(m, x)
    compact_support = (-200, 200)
    compute_sums(beta, w) = gs.compute_sums_generic(phi_hat, compact_support, beta, w)
    
    I = length(alpha_zz)

    A_gs = T1[]
    B_gs = T1[]
    alpha_gs = T1[]
    beta_gs = T1[]

    for (i, (alpha_i, beta_i)) in enumertae(zip(alpha_zz, beta_zz))
        if print_progress > 0
            println("Iteration (alpha, beta) = ($alpha_i, $beta_i) ($i out of $I)")
        end
        Ai, Bi, _= gs.frame_bounds_fixed_beta(compute_sums, [alpha_i], beta_i, dw; B_symmetric, optim_iter_M, optim_iter_s2, T2)

        if length(Ai) > 0
            append!(A_gs, Ai)
            append!(B_gs, Bi)
            append!(alpha_gs, alpha_i)
            append!(beta_gs, beta_i)
        end
    end
    save_bounds(gs_path, A_gs, B_gs, alpha_gs, beta_gs)

else
    A_zz, B_zz, alpha_zz, beta_zz = load_bounds(zz_path)
    A_gs, B_gs, alpha_gs, beta_gs = load_bounds(gs_path)

end

is_frame_gs = is_gabor_frame(A_gs, B_gs)
is_frame_zz = is_gabor_frame(A_zz, B_zz, min_tol = 1e-20)

is_frame_subplot_gs = is_gabor_frame(A_gs, B_gs, min_tol = 1e0)
is_frame_subplot_zz = is_gabor_frame(A_zz, B_zz, min_tol = 1e0)


### Plot bounds along hyperbola
plt = scatter(beta_gs[is_frame_gs], B_gs[is_frame_gs], markersize=1, markerstrokewidth=0, label="GS upper bound", yscale =:log10, legend=:bottomleft)
scatter!(beta_gs[is_frame_gs], A_gs[is_frame_gs], markersize=1, markerstrokewidth=0, label="GS lower bound")
scatter!(beta_zz[is_frame_zz], B_zz[is_frame_zz], markersize=1, markerstrokewidth=0, label="ZZ upper bound")
scatter!(beta_zz[is_frame_zz], A_zz[is_frame_zz], markersize=1, markerstrokewidth=0, label="ZZ lower bound")

plot!(plt, inset=bbox(0.0, 0.15, 0.3, 0.25, :top, :right), subplot=2, legend=false, yscale =:log10)
scatter!(plt[2], beta_gs[is_frame_subplot_gs], [B_gs[is_frame_subplot_gs], A_gs[is_frame_subplot_gs]],  markersize=1, markerstrokewidth=0)
scatter!(plt[2], beta_zz[is_frame_subplot_zz], [B_zz[is_frame_subplot_zz], A_zz[is_frame_subplot_zz]],  markersize=1, markerstrokewidth=0)


xlabel!("\\beta")
display(plt)
savefig(plt, "plots/bspline_order_$(m)_hyperbola_bounds.png")