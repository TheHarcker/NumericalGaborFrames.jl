
include("../utils.jl")
include("../modules/ghosh_selvan.jl")
include("../modules/ghosh_selvan_guaranteed_bound.jl")
import .GhoshSelvan as gs

function plot_guaranteed_bound(m::Integer, num_q::Integer, alpha_max::Real, beta_min::Integer, beta_max::Integer; show_as_ab_b_plot::Bool=false, make_plot::Bool = true, binary_search_for_d_with_iterations::Integer = 10)
    @assert num_q > 0 && m > 0 && alpha_max > 0 && beta_max >= beta_min> 0
    T1 = Float64
    alphas_omega2::Vector{T1} = []
    alphas_omega1::Vector{T1} = []
    betas::Vector{T1} = []

    for q in 1:num_q
        (new_non_periodic_alphas_omega2, new_betas) = GhoshSelvanGuaranteedBound.NonPeriodicBound.get_points_non_periodic_bound(q, m, beta_max; binary_search_for_d_with_iterations)
        alphas_omega2 = vcat(alphas_omega2, new_non_periodic_alphas_omega2) 
        (new_non_periodic_alphas_omega1, _) = GhoshSelvanGuaranteedBound.NonPeriodicBound.get_points_non_periodic_bound(q, m, beta_max; binary_search_for_d_with_iterations, use_omega_1_as_numerator=true)
        alphas_omega1 = vcat(alphas_omega1, new_non_periodic_alphas_omega1) 

        betas = vcat(betas, new_betas)
        println("Iteration q=", q, " ", q/num_q * 100, "%")
    end

    marzieh_betas = collect(LinRange(beta_min, beta_max, (beta_max - beta_min) * 1000))
    marzieh_alphas = GhoshSelvanGuaranteedBound.MarziehBound.get_points(marzieh_betas, m)

    compute_sums(beta, w) = gs.compute_sums_bspline(m, beta, w)

    dw = 0.001
    gs_betas = collect(LinRange(beta_min, beta_max, (beta_max - beta_min) * 100))
    gs_alphas = gs.frame_set_max_alpha(compute_sums, gs_betas, dw; B_symmetric= true, parallelize=true)

    indices = sortperm(gs_betas)
    @assert length(gs_alphas) == length(gs_betas) == length(indices)
    gs_alphas = gs_alphas[indices]
    gs_betas = gs_betas[indices]

    # Show as (ab, b) plot
    if show_as_ab_b_plot
        gs_alphas .*= gs_betas
        alphas_omega1 .*= betas
        alphas_omega2 .*= betas
        marzieh_alphas .*= marzieh_betas
    end 

    # Plot bounds for Q_m(x)
    if make_plot
        if show_as_ab_b_plot
            combined_plot = plot(gs_alphas, gs_betas, lw=1, color="yellow", label="Ghosh-Selvan", fillrange=-9)
        else
            gs_alphas[gs_alphas .<= 1e-9] .= 1e-9 # Avoid nan values 
            combined_plot = plot(gs_alphas, gs_betas, lw=1, color="yellow", label="Ghosh-Selvan", fillrange=-9, xscale = :log10)
        end
        plot!(marzieh_alphas, marzieh_betas, lw=1, color="green", label="Trivial bound", fillrange=-9)
        
        scatter!(alphas_omega1[betas .<= beta_max], betas[betas .<= beta_max], color="blue", label="\\alpha_{max} with \\Omega_1", markersize = 3)
        scatter!(alphas_omega2[betas .<= beta_max], betas[betas .<= beta_max], color="pink", label="\\alpha_{max} with \\Omega_2", markersize = 3)
        xlabel!(show_as_ab_b_plot ? "\\alpha\\beta" : "\\alpha")
        ylabel!("\\beta")
        yticks!(beta_min:beta_max)
        xlims!(1e-9, min(alpha_max, 1/beta_min, m/2))
        ylims!(beta_min, beta_max)
        title!("Frame set for Q_$m")
        display(combined_plot)
        savefig(combined_plot, "Figures/julia/gs_guaranteed_$(show_as_ab_b_plot ? "ab-b" : "a-b")_$m$(GhoshSelvanGuaranteedBound.Constants.assume_sigma_inf_conjecture ? "_convex" : "_non-convex").png")
    end
end
plot_guaranteed_bound(4, 100, 1, 1, 4; show_as_ab_b_plot=false, binary_search_for_d_with_iterations=10)

