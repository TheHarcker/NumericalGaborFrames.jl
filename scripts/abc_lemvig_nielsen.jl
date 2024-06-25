include("../utils.jl")
include("../modules/ghosh_selvan.jl")

import .GhoshSelvan as gs

compute = false

T1 = Float64
T2 = Double64

optim_iter_M = 20
print_progress = 1
B_symmetric = true

m = 2
phi_hat = x -> bspline_hat(m, x)
compact_support = (-200, 200)
compute_sums(beta, w) = gs.compute_sums_generic(phi_hat, compact_support, beta, w)

fractions = generate_reduced_fractions_below_one(1, 10)
beta_min = 0
beta_max = 10
dbeta = 0.01
dw = 0.001

function lemvig_nielsen(
    m::Integer,
    fractions::Vector{<:Rational}, 
    alpha::Vector{T1}, 
    beta::Vector{T1};
    alpha_max::Real = Inf, 
    beta_max::Real = Inf, 
    alpha_min::Real = 0, 
    beta_min::Real = 0, 
)::Tuple{BitVector, Vector{T1}, Vector{T1}} where T1<:Real
    f = is_not_frame_rationally_oversampled_bspline(m)
    is_not_frame = []
    alpha_new = T1[]
    beta_new = T1[]

    for frac in fractions
        pk, qk = numerator(frac), denominator(frac)

        alpha_frac = (pk / qk) ./ beta
        beta_frac = (pk / qk) ./ alpha

        alpha_beta = vcat(collect(zip(alpha_frac, beta)), collect(zip(alpha, beta_frac)))
        filter!(tuple -> alpha_min < tuple[1] < alpha_max && beta_min < tuple[2] < beta_max, alpha_beta)

        for (alpha_j, beta_j) in alpha_beta
            push!(is_not_frame, f(alpha_j, beta_j, pk, qk))            
            push!(alpha_new, alpha_j)
            push!(beta_new, beta_j)
        end
    end

    return is_not_frame, alpha_new, beta_new
end

file_path_gs = "data/frame_sets/abc_lemvig_nielsen_bspline_$m.npz"
if compute
    beta_gs = uniform_spaced_values(dbeta, beta_max, dbeta; T=T1)
    alpha_gs = gs.frame_set_max_alpha(compute_sums, beta_gs, dw; optim_iter_M, B_symmetric, print_progress, T2)
    alpha_gs .*= beta_gs

    save_alpha_beta(file_path_gs, alpha_gs, beta_gs)
else
    alpha_gs, beta_gs = load_alpha_beta(file_path_gs)
end

beta_ce = []
alpha_ce = []
if m == 2
    counter_examples = [(2,1), (3,2), (4,3), (5,4), (6,5), (7,6), (8,7), (9,8)]
    for (i, (k, m)) in enumerate(counter_examples)
        a0 = 1/(2*m + 1)
        b0 = (2*k + 1)/2
        append!(beta_ce, uniform_spaced_values(b0- a0*(k-m)/2, b0 + a0*(k-m)/2, 0.01))
        append!(alpha_ce, (2*k+1)./(2*(2*m+1).*uniform_spaced_values(b0- a0*(k-m)/2, b0 + a0*(k-m)/2, 0.01)))
    end
end
alpha_ce .*= beta_ce

is_not_frame, alpha, beta = lemvig_nielsen(m, fractions, T1[], beta_gs; beta_max)
alpha .*= beta

plt = plot(beta_gs, alpha_gs, label = "GS max alpha")
scatter!(beta[is_not_frame], alpha[is_not_frame], markersize=1, markerstrokewidth=0, label = "Lemvig Nielsen is not frame", c=:red)
scatter!(beta_ce, alpha_ce, markersize = 1, markerstrokewidth = 0, c=:red, label = false)
xlabel!("\\beta")
ylabel!("\\alpha \\beta")
display(plt)
savefig(plt, "plots/abc_lemvig_nielsen_bspline_$m.png")