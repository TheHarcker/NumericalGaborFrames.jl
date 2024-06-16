# Run if script is executed directly
include("../utils.jl")
include("../modules/ghosh_selvan.jl")
include("../modules/zibulski_zeevi.jl")

import .ZibulskiZeevi as zz
import .GhoshSelvan as gs

# --- Symbolic hermite functions in Julia ---
# using Symbolics
# function hermite(n)
#    @variables x
#    D = Differential(x)#
#    cn = (2*pi)^n * 2^(n - 1/2) * factorial(n)
#    hn = (-1)^n * (cn)^(-1/2) * exp(pi * x^2) * expand_derivatives((D^n)(exp(-2 * pi * x^2)))
#    return build_function(hn, x; expression = Val{false})
# end

hermite_2(x) = 2^(3/4)*(4*pi*x^2 - 1)*exp(-pi*x^2)/2
hermite_3(x) = ((4*pi*x^3 - 3*x)*sqrt(3)*2^(3/4)*sqrt(pi)*exp(-pi*x^2))/3
hermite_4(x) = 2^(3/4)*sqrt(3)*exp(-pi*x^2)*(16*pi^2*x^4 - 24*pi*x^2 + 3)/12
hermite_5(x) = ((16*pi^2*x^5 - 40*pi*x^3 + 15*x)*2^(3/4)*sqrt(15)*sqrt(pi)*exp(-pi*x^2))/30

#setprecision(BigFloat, 256)
compute = true
T1 = Float64
T2 = Double64
print_progress = 2
optim_iter_M = 20
optim_iter_s2 = 20
optim_iter = 20

alpha_max = 5
beta_max = 5
dalpha = 0.01
dbeta = 0.01
dw = 0.001

compact_support = (-10, 10)
fractions = generate_reduced_fractions_below_one(5, 5)
dt = 0.01
dv = 0.01

plot_hermite_functions = true
if plot_hermite_functions
    dx = 0.01
    x = uniform_spaced_values(-3, 3, dx)

    plt = plot(x, hermite_2.(x), label="Hermite 2")
    plot!(x, hermite_3.(x), label="Hermite 3")
    plot!(x, hermite_4.(x), label="Hermite 4")
    plot!(x, hermite_5.(x), label="Hermite 5")
    title!("Hermite functions")
    xlabel!("x")
    ylabel!("y")
    display(plt)
    savefig(plt, "../Figures/julia/hermite_functions.png")
end

for (m, hn) in zip([2,3,4,5], [hermite_2, hermite_3, hermite_4, hermite_5])
    if compute
        phi(x) = hn.(x)
        compute_sums(beta, w) = gs.compute_sums_generic(phi, compact_support, beta, w)

        A_gs, B_gs, alpha_gs, beta_gs = gs.frame_bounds_grid(compute_sums, dalpha, dbeta, dw, alpha_max, beta_max; optim_iter_M, optim_iter_s2, print_progress, T1, T2)
        A_zz, B_zz, alpha_zz, beta_zz = zz.frame_bounds_grid(phi, compact_support, fractions, dalpha, dbeta, dt, dv, alpha_max, beta_max; optim_iter, print_progress, T1, T2)

        save_bounds("../Data/frame_bounds/gs_hermite_$m.npz", A_gs, B_gs, alpha_gs, beta_gs)
        save_bounds("../Data/frame_bounds/zz_hermite_$m.npz", A_zz, B_zz, alpha_zz, beta_zz)
    else
        A_gs, B_gs, alpha_gs, beta_gs = load_bounds("../Data/frame_bounds/gs_hermite_$m.npz")
        A_zz, B_zz, alpha_zz, beta_zz = load_bounds("../Data/frame_bounds/zz_hermite_$m.npz")
    end
        
    is_frame_gs = is_gabor_frame(A_gs, B_gs)
    is_frame_zz = is_gabor_frame(A_zz, B_zz, min_tol=1e-20)

    plt = scatter(alpha_gs[is_frame_gs], beta_gs[is_frame_gs], markersize=1, markerstrokewidth=0, label="GS is frame", alpha=0.5)
    scatter!(beta_gs[is_frame_gs], alpha_gs[is_frame_gs], markersize=1, markerstrokewidth=0, label="GS is frame", alpha=0.5)
    scatter!(alpha_zz[is_frame_zz], beta_zz[is_frame_zz], markersize=1, markerstrokewidth=0, label="ZZ is frame", alpha=0.5)
    scatter!(alpha_zz[.!is_frame_zz], beta_zz[.!is_frame_zz], markersize=1, markerstrokewidth=0, label="ZZ is not frame", alpha=0.5)
    #title!("h_$m frame set (T = $T2, dw = $dw, dv = $dv, dt = $dt)")
    xlabel!("\\alpha")
    ylabel!("\\beta")
    display(plt)
    savefig(plt, "../Figures/julia/frame_set_hermite_$m.png")

    plt = scatter3d(alpha_gs[is_frame_gs], beta_gs[is_frame_gs], log10.(A_gs[is_frame_gs]), label="GS lower bound", markersize=1, markerstrokewidth=0, camera=(135,30), alpha=0.5)
    scatter3d!(alpha_gs[is_frame_gs], beta_gs[is_frame_gs], log10.(B_gs[is_frame_gs]), label="GS upper bound", markersize=1, markerstrokewidth=0, alpha=0.5)
    scatter3d!(beta_gs[is_frame_gs], alpha_gs[is_frame_gs], log10.(A_gs[is_frame_gs]), label="GS lower bound", markersize=1, markerstrokewidth=0, alpha=0.5)
    scatter3d!(beta_gs[is_frame_gs], alpha_gs[is_frame_gs], log10.(B_gs[is_frame_gs]), label="GS upper bound", markersize=1, markerstrokewidth=0, alpha=0.5)
    scatter3d!(alpha_zz[is_frame_zz], beta_zz[is_frame_zz], log10.(A_zz[is_frame_zz]), label="ZZ lower bound", markersize=1, markerstrokewidth=0, alpha=0.5)
    scatter3d!(alpha_zz[is_frame_zz], beta_zz[is_frame_zz], log10.(B_zz[is_frame_zz]), label="ZZ upper bound", markersize=1, markerstrokewidth=0, alpha=0.5)
    #title!("h_$m frame bounds (T = $T2, dw = $dw, dv = $dv, dt = $dt)")
    xlabel!("\\alpha")
    ylabel!("\\beta")
    display(plt)
    savefig(plt, "../Figures/julia/frame_bounds_hermite_$m.png")
end