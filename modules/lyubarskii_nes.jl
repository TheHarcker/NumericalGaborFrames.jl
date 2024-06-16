module LyubarskiiNes
export frame_set, frame_set_grid

include("../utils.jl")

function compute_transform(
    phi::Function, 
    compact_support::Tuple{Real, Real}, 
    alpha::Real, 
    q::Integer, 
    t::Vector{T}, 
    v::Vector{T}, 
)::Matrix{<:Complex} where T<:Real
    a, b = compact_support
    @assert a < b && alpha > 0 && q > 0
    
    kmin = floor(Int, minimum((t .- b)/(alpha * q)))
    kmax = ceil(Int, maximum((t .- a)/(alpha * q)))

    z = sum(phi(t .- alpha * q * n) * exp.((2 * pi * im * alpha * q * n) .* v)' for n in kmin:kmax)

    @assert (length(t), length(v)) == size(z)
    return z
end

function compute_matrix(
    phi::Function, 
    compact_support::Tuple{Real, Real}, 
    alpha::Real, 
    beta::Real, 
    p::Integer, 
    q::Integer, 
    t::Vector{T}, 
    v::Vector{T},
)::Array{<:Complex, 4} where T<:Real
    @assert 0 < p < q && beta > 0

    Phi = stack([compute_transform(phi, compact_support, alpha, q, t .+ (alpha*l + k/beta), v) for k in 0:(p-1), l in 0:(q-1)])

    @assert size(Phi) == (length(t), length(v), p, q)
    return Phi
end

function matrix_svd_ratio(A::Matrix{<:Complex})::Real
    S = svd(A).S
    return S[end]/S[1]
end

function compute_svd_ratio(
    phi::Function, 
    compact_support::Tuple{Real, Real}, 
    alpha::T1, 
    beta::T1, 
    p::Integer, 
    q::Integer, 
    dt::Real, 
    dv::Real; 
    optim_iter::Integer = 0,
    T2::Type{<:Real} = Float64, 
)::T1 where T1<:Real
    @assert abs(alpha*beta - p/q) < 1e-10 && dt > 0 && dv > 0 && optim_iter >= 0

    t = uniform_spaced_values(0, alpha/p, dt; T=T2)
    v = uniform_spaced_values(0, 1/alpha, dv; T=T2)
    Phi = compute_matrix(phi, compact_support, alpha, beta, p, q, t, v)

    R = map(A -> matrix_svd_ratio(A), Phi[i, j, :, :] for i in axes(Phi, 1), j in axes(Phi, 2))
    (R_min, idx_R_min) = findmin(R)

    if optim_iter > 0
        f(x) = svdvals(compute_matrix(phi, compact_support, alpha, beta, p, q, [x[1]], [x[2]])[1, 1, :, :]) |> S -> S[end]/S[1]
        R_min = minimize_unbounded(f, [t[idx_R_min[1]], v[idx_R_min[2]]], R_min, optim_iter)
    end

    @assert size(R) == (length(t), length(v))
    return R_min
end

function frame_set(
    phi::Function, 
    compact_support::Tuple{Real, Real}, 
    fractions::Vector{Rational}, 
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
    print_progress_layer::Integer = 0,
    T2::Type{<:Real} = Float64, 
)::Tuple{Vector{T1}, Vector{T1}, Vector{T1}} where T1<:Real
    @assert 0 <= alpha_min <= alpha_max && 0 <= beta_min < beta_max && print_progress_layer >= 0

    svd_ratio = T1[]
    alpha_new = T1[]
    beta_new = T1[]

    K = length(fractions)
    for (k, frac) in enumerate(fractions)
        pk, qk = numerator(frac), denominator(frac)

        if print_progress_layer > 0
            println("Iteration (p, q) = ($pk, $qk) ($k out of $K)")
        end

        alpha_frac = (pk / qk) ./ beta
        beta_frac = (pk / qk) ./ alpha

        alpha_beta = vcat(collect(zip(alpha_frac, beta)), collect(zip(alpha, beta_frac)))
        filter!(tuple -> alpha_min < tuple[1] < alpha_max && beta_min < tuple[2] < beta_max, alpha_beta)

        J = length(alpha_beta)
        for (j, (alpha_j, beta_j)) in enumerate(alpha_beta)
            if print_progress_layer > 1
                println("    Iteration (alpha, beta) = ($alpha_j, $beta_j) ($j out of $J)")
            end

            if !isnothing(is_not_frame) && is_not_frame(alpha_j, beta_j, pk, qk)
                svd_ratio_j = NaN
            else
                svd_ratio_j = compute_svd_ratio(phi, compact_support, alpha_j, beta_j, pk, qk, dt, dv; optim_iter, T2)
            end

            push!(svd_ratio, svd_ratio_j)
            push!(alpha_new, alpha_j)
            push!(beta_new, beta_j)
        end
    end

    @assert length(svd_ratio) == length(alpha_new) == length(beta_new)
    return svd_ratio, alpha_new, beta_new
end

function frame_set_grid(
    phi::Function, 
    compact_support::Tuple{Real, Real}, 
    fractions::Vector{Rational}, 
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
    print_progress_layer::Integer = 0,
    T1::Type{<:Real} = Float64, 
    T2::Type{<:Real} = Float64, 
)::Tuple{Vector{T1}, Vector{T1}, Vector{T1}}
    alpha = uniform_spaced_values(alpha_min > 0 ? alpha_min : dalpha, alpha_max, dalpha; T=T1)
    beta = uniform_spaced_values(beta_min > 0 ? beta_min : dbeta, beta_max, dbeta; T=T1)

    return frame_set(phi, compact_support, fractions, alpha, beta, dt, dv; alpha_max, beta_max, alpha_min, beta_min, optim_iter, is_not_frame, print_progress_layer, T2)
end

end
