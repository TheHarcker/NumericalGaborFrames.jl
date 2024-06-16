module ZibulskiZeevi
export frame_bounds, frame_bounds_grid

include("../utils.jl")

function compute_zak_transform(
    phi::Function, 
    compact_support::Tuple{Real, Real}, 
    lamb::Real, 
    t::Vector{T}, 
    v::Vector{T}
)::Matrix{<:Complex} where T<:Real
    a, b = compact_support
    @assert lamb > 0 && a < b

    kmin = floor(Int, minimum(t) - b/lamb)
    kmax = ceil(Int, maximum(t) - a/lamb)

    z = sqrt(lamb) * sum(phi(lamb * (t .- k)) * exp.((2 * pi * im * k) * v)' for k in kmin:kmax)

    @assert size(z) == (length(t), length(v))
    return z
end

function compute_zibulsk_zeevi_matrix(
    phi::Function, 
    compact_support::Tuple{Real, Real}, 
    beta::Real, 
    p::Integer, 
    q::Integer, 
    t::Vector{T}, 
    v::Vector{T}
)::Array{<:Complex, 4} where T<:Real
    @assert 0 < p < q && beta > 0

    Phi = stack([compute_zak_transform(phi, compact_support, 1/beta, t .- l * p/q, v .+ k/p) for k in 0:p-1, l in 0:q-1] / sqrt(p))

    @assert size(Phi) == (length(t), length(v), p, q)
    return Phi
end

function compute_frame_bounds(
    phi::Function, 
    compact_support::Tuple{Real, Real}, 
    alpha::T1, 
    beta::T1, 
    p::Integer, 
    q::Integer, 
    t::Vector{T2}, 
    v::Vector{T2}; 
    optim_iter::Integer = 0
)::Tuple{T1, T1} where {T1<:Real, T2<:Real}
    @assert abs(alpha*beta - p/q) < 1e-10 && optim_iter >= 0

    Phi = compute_zibulsk_zeevi_matrix(phi, compact_support, beta, p, q, t, v)
    S = map(A -> svdvals(A), Phi[i, j, :, :] for i in axes(Phi, 1), j in axes(Phi, 2))
    (A, idx_A) = findmin(map(v -> v[end], S))
    (B, idx_B) = findmax(map(v -> v[1], S))

    if optim_iter > 0
        f(x) = svdvals(compute_zibulsk_zeevi_matrix(phi, compact_support, beta, p, q, [x[1]], [x[2]])[1, 1, :, :])

        A = minimize_unbounded(x -> f(x) |> last,  [t[idx_A[1]], v[idx_A[2]]], A, optim_iter)
        B = maximize_unbounded(x -> f(x) |> first, [t[idx_B[1]], v[idx_B[2]]], B, optim_iter)
    end

    @assert size(S) == (length(t), length(v))
    return T1(A^2), T1(B^2)
end

function frame_bounds(
    phi::Function, 
    compact_support::Tuple{Real, Real}, 
    fractions::Vector{<:Rational}, 
    alpha::Vector{T1}, 
    beta::Vector{T1}, 
    dt::Real, 
    dv::Real; 
    alpha_max::Real = Inf, 
    beta_max::Real = Inf, 
    alpha_min::Real = 0, 
    beta_min::Real = 0, 
    optim_iter::Integer = 0,
    is_not_frame::Union{Function, Nothing} = nothing,
    print_progress::Integer = 0,
    T2::Type{<:Real} = Float64, 
)::Tuple{Vector{T1}, Vector{T1}, Vector{T1}, Vector{T1}} where T1<:Real
    @assert print_progress >= 0 && 0 <= alpha_min <= alpha_max && 0 <= beta_min <= beta_max && dt > 0 && dv > 0

    A = T1[]
    B = T1[]
    alpha_new = T1[]
    beta_new = T1[]

    t = uniform_spaced_values(0, 1-dt, dt; T=T2)
    v = uniform_spaced_values(0, 1-dv, dv; T=T2)

    K = length(fractions)
    for (k, frac) in enumerate(fractions)
        pk, qk = numerator(frac), denominator(frac)

        if print_progress > 0
            println("Iteration (p, q) = ($pk, $qk) ($k out of $K)")
        end

        alpha_frac = (pk / qk) ./ beta
        beta_frac = (pk / qk) ./ alpha

        alpha_beta = vcat(collect(zip(alpha_frac, beta)), collect(zip(alpha, beta_frac)))
        filter!(tuple -> alpha_min < tuple[1] < alpha_max && beta_min < tuple[2] < beta_max, alpha_beta)

        J = length(alpha_beta)
        for (j, (alpha_j, beta_j)) in enumerate(alpha_beta)
            if print_progress > 1
                println("    Iteration (alpha, beta) = ($alpha_j, $beta_j) ($j out of $J)")
            end

            if !isnothing(is_not_frame) && is_not_frame(alpha_j, beta_j, pk, qk)
                Aj, Bj = NaN, NaN
            else
                Aj, Bj = compute_frame_bounds(phi, compact_support, alpha_j, beta_j, pk, qk, t, v; optim_iter)
            end
            
            push!(A, Aj)
            push!(B, Bj)
            push!(alpha_new, alpha_j)
            push!(beta_new, beta_j)
        end
    end

    @assert length(A) == length(B) == length(alpha_new) == length(beta_new)
    return A, B, alpha_new, beta_new
end

function frame_bounds_grid(
    phi::Function, 
    compact_support::Tuple{Real, Real}, 
    fractions::Vector{<:Rational}, 
    dalpha::Real, 
    dbeta::Real, 
    dt::Real, 
    dv::Real, 
    alpha_max::Real, 
    beta_max::Real; 
    alpha_min::Real = 0, 
    beta_min::Real = 0, 
    optim_iter::Integer = 0,
    is_not_frame::Union{Function, Nothing} = nothing, 
    print_progress::Integer = 0,
    T1::Type{<:Real} = Float64,
    T2::Type{<:Real} = Float64,
)::Tuple{Vector{<:Real}, Vector{<:Real}, Vector{<:Real}, Vector{<:Real}}
    alpha = uniform_spaced_values(alpha_min > 0 ? alpha_min : dalpha, alpha_max, dalpha; T=T1)
    beta = uniform_spaced_values(beta_min > 0 ? beta_min : dbeta, beta_max, dbeta; T=T1)

    return frame_bounds(phi, compact_support, fractions, alpha, beta, dt, dv; alpha_max, beta_max, alpha_min, beta_min, optim_iter, is_not_frame, print_progress, T2)
end

end