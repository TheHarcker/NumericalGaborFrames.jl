module GhoshSelvan
export compute_sums_bspline, compute_sums_generic, frame_set_max_alpha, frame_bounds_fixed_beta, frame_bounds_grid

include("../utils.jl")

function compute_sums_generic(
    phi_hat::Function, 
    compact_support::Tuple{Real, Real}, 
    beta::Real, 
    w::Vector{T}
)::Tuple{Vector{T}, Vector{T}} where T<:Real
    a, b = compact_support
    @assert beta > 0 && a < b

    kmin = floor(Int, (a - maximum(w)) / beta)
    kmax = ceil(Int, (b - minimum(w)) / beta)

    s1 = sum((w .+ k*beta).^2 .* abs2.(phi_hat(w .+ k*beta)) for k in kmin:kmax)
    s2 = sum(abs2.(phi_hat(w .+ k*beta)) for k in kmin:kmax)

    @assert size(w) == size(s1) == size(s2)
    return s1, s2
end

function compute_sums_bspline(
    m::Integer, 
    beta::Real, 
    w::Vector{T}
)::Tuple{Vector{T}, Vector{T}} where T<:Real
    @assert m > 0 && beta > 0

    n = collect(T, 1:ceil(Int, m * beta))
    x = n ./ beta
    cos_vals = cos.(((2 * pi / beta) .* w) * n')

    s1 = vec(((bspline(2*m-2, T(0)) - bspline(2*m-2, T(1))) .+ sum(cos_vals .* (2*bspline(2*m-2, x) 
        - bspline(2*m-2, x .+ 1) - bspline(2*m-2, x .- 1))', dims=2)) / (2 * pi^2 * beta))
    s2 = vec((bspline(2*m, T(0)) .+ 2*sum(cos_vals .* bspline(2*m, x)', dims=2)) / beta)

    @assert size(w) == size(s1) == size(s2)
    return s1, s2
end

function compute_M(
    compute_sums::Function, 
    beta::Real, 
    dw::Real;
    B_symmetric::Bool = false,
    ensure_integer_w::Bool = true,
    tol_M::Real = 1e-10,
    optim_iter_M::Integer = 0,
    T2::Type{<:Real} = Float64, 
)::Tuple{T2, Vector{T2}, Vector{T2}}
    @assert optim_iter_M >= 0

    b = B_symmetric ? beta / 2 : beta 
    w = uniform_spaced_values(0, b, dw; T=T2)
    
    if ensure_integer_w && beta > 1
        w = vcat(w, 1:floor(beta))
    end

    s1, s2 = compute_sums(beta, w)
    (M, idx_M) = findmax(s1 ./ s2)

    if optim_iter_M > 0
        f(w) = compute_sums(beta, w) |> x -> x[1][1] / x[2][1]
        M = maximize_bounded(f, [T2(0)], [T2(b)], [w[idx_M]], M, optim_iter_M)
    end

    if beta^2 >= 4*M + tol_M
        M = NaN
    end

    @assert size(w) == size(s2)
    return M, w, s2
end

function compute_min_max_s2(
    compute_sums::Function, 
    beta::Real, 
    w::Vector{T2},
    s2::Vector{T2};
    optim_iter_s2::Integer = 0
)::Tuple{T2, T2} where T2<:Real
    @assert optim_iter_s2 >= 0 && size(w) == size(s2)

    (s2_min, idx_s2_min) = findmin(s2)
    (s2_max, idx_s2_max) = findmax(s2)

    if optim_iter_s2 > 0
        f(w) = compute_sums(beta, w)[2][1]
        b = w[end]

        s2_min = minimize_bounded(f, [T2(0)], [T2(b)], [w[idx_s2_min]], s2_min, optim_iter_s2)
        s2_max = maximize_bounded(f, [T2(0)], [T2(b)], [w[idx_s2_max]], s2_max, optim_iter_s2)
    end

    return s2_min, s2_max
end

function frame_set_max_alpha(
    compute_sums::Function, 
    beta::Vector{T1}, 
    dw::Real; 
    B_symmetric::Bool = false,
    ensure_integer_w::Bool = true,
    tol_M::Real = 1e-10,
    optim_iter_M::Integer = 0,
    print_progress::Integer = 0, 
    parallelize::Bool = false, 
    T2::Type{<:Real} = Float64, 
)::Vector{T1} where T1<:Real
    @assert print_progress >= 0

    alpha_max = zeros(length(beta))

    I = length(beta)
    function f(i, beta_i)
        if print_progress > 0
            println("Iteration beta = $(beta_i) ($i out of $I)")
        end

        M, _, _ = compute_M(compute_sums, beta_i, dw; B_symmetric, ensure_integer_w, tol_M, optim_iter_M, T2)

        if isfinite(M) && M > 0
            alpha_max[i] = 1/(2*sqrt(M))
        end
    end 

    if parallelize
        Threads.@threads for i in axes(beta, 1)
            f(i, beta[i])
        end
    else
        for (i, beta_i) in enumerate(beta) 
            f(i, beta_i)
        end
    end 

    @assert size(alpha_max) == size(beta)
    return alpha_max
end

function frame_bounds_fixed_beta(
    compute_sums::Function, 
    alpha::Vector{T1}, 
    beta::Real, 
    dw::Real; 
    B_symmetric::Bool = false,
    ensure_integer_w::Bool = true,
    tol_M::Real = 1e-10,
    optim_iter_M::Integer = 0, 
    optim_iter_s2::Integer = 0, 
    T2::Type{<:Real} = Float64,
)::Tuple{Vector{T1}, Vector{T1}, Vector{T1}} where T1<:Real
    M, w, s2 = compute_M(compute_sums, beta, dw; B_symmetric, ensure_integer_w, tol_M, optim_iter_M, T2)

    if isfinite(M) && M > 0
        alpha = alpha[alpha .< 1/(2*sqrt(M))]
        s2_min, s2_max = compute_min_max_s2(compute_sums, beta, w, s2; optim_iter_s2)

        c = 2 * sqrt(M)
        A = Vector{T1}((1 .- c .* alpha).^2 .* s2_min ./ alpha)
        B = Vector{T1}((1 .+ c .* alpha).^2 .* s2_max ./ alpha)
    else
        A, B, alpha = T1[], T1[], T1[]
    end

    @assert size(A) == size(B) == size(alpha)
    return A, B, alpha
end

function frame_bounds(
    compute_sums::Function, 
    alpha::Vector{T1}, 
    beta::Vector{T1}, 
    dw::Real;
    B_symmetric::Bool = false,
    ensure_integer_w::Bool = true,
    tol_M::Real = 1e-10,
    optim_iter_M::Integer = 0, 
    optim_iter_s2::Integer = 0, 
    print_progress::Integer = 0,
    T2::Type{<:Real} = Float64,
)::Tuple{Vector{T1}, Vector{T1}, Vector{T1}, Vector{T1}} where T1<:Real
    A = T1[]
    B = T1[]
    alpha_new = T1[]
    beta_new = T1[]

    I = length(beta)
    for (i, beta_i) in enumerate(beta)
        if print_progress > 0
            println("Iteration beta = $(beta_i) ($i out of $I)")
        end

        Ai, Bi, alpha_i = frame_bounds_fixed_beta(compute_sums, alpha, beta_i, dw; B_symmetric, ensure_integer_w, tol_M, optim_iter_M, optim_iter_s2, T2)
        N = length(Ai)

        if N > 0
            append!(A, Ai)
            append!(B, Bi)
            append!(alpha_new, alpha_i)
            append!(beta_new, fill(beta_i, N))
        end
    end

    @assert size(A) == size(B) == size(alpha_new) == size(beta_new)
    return A, B, alpha_new, beta_new
end

function frame_bounds_grid(
    compute_sums::Function, 
    dalpha::Real, 
    dbeta::Real, 
    dw::Real, 
    alpha_max::Real, 
    beta_max::Real; 
    alpha_min::Real = dalpha, 
    beta_min::Real = dbeta,
    B_symmetric::Bool = false,
    ensure_integer_w::Bool = true,
    tol_M::Real = 1e-10,
    optim_iter_M::Integer = 0, 
    optim_iter_s2::Integer = 0, 
    print_progress::Integer = 0,
    T1::Type{<:Real} = Float64,
    T2::Type{<:Real} = Float64,
)::Tuple{Vector{T1}, Vector{T1}, Vector{T1}, Vector{T1}} 
    alpha = uniform_spaced_values(alpha_min, alpha_max, dalpha; T=T1)
    beta = uniform_spaced_values(beta_min, beta_max, dbeta; T=T1)

    return frame_bounds(compute_sums, alpha, beta, dw; B_symmetric, ensure_integer_w, tol_M, optim_iter_M, optim_iter_s2, print_progress, T2)
end

end