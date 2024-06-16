using LinearAlgebra
using Plots
using DoubleFloats
using Optim
using NPZ
import CSV

function uniform_spaced_values(
    x1::Real, 
    xn::Real, 
    dx::Real; 
    T::Type{<:Real} = Float64
)::Vector{T}
    @assert dx > 0 && x1 < xn

    n = ceil(Int, (xn - x1) / dx) + 1
    return collect(T, range(x1, stop=xn, length=n))
end

function bspline(m::Integer, x::T)::T where T<:Real
    @assert m > 0

    y = typeof(x)((-m/2 <= x) & (x <= m/2))
    if m >= 2
        y *= sum((-1)^j * binomial(m, j) * max(x + m/2 - j, 0)^(m - 1) for j in 0:m) / factorial(m - 1)
    end

    return y
end

function bspline(m::Integer, x::Vector{T})::Vector{T} where T<:Real
    @assert m > 0

    y = typeof(x)((-m/2 .<= x) .& (x .<= m/2))
    if m >= 2
        y .*= sum((-1)^j * binomial(m, j) * max.(x .+ (m/2 - j), 0).^(m - 1) for j in 0:m) / factorial(m - 1)
    end

    @assert size(x) ==  size(y)
    return y
end

function bspline_hat(m::Integer, x::T)::T where T<:Real
    @assert m > 0

    if x == 0
        return 1
    end

    return (sin(pi * x) / (pi * x))^m
end

function bspline_hat(m::Integer, x::Vector{T})::Vector{T} where T<:Real
    return bspline_hat.(m, x)
end

function minimize_unbounded(
    f::Function, 
    x0::Vector{T}, 
    y0::T, 
    optim_iter::Integer; 
    method::M = LBFGS(),
    print_error::Bool = true,
)::T where {T<:Real, M<:Optim.AbstractOptimizer}
    @assert optim_iter > 0

    try
        y = minimum(optimize(f, x0, method, Optim.Options(iterations = optim_iter)))
        return min(y, y0)
    catch e
        if print_error
            println("Optim error: ", e)
        end
        return y0
    end
end

function maximize_unbounded(
    f::Function, 
    x0::Vector{T}, 
    y0::T, 
    optim_iter::Integer; 
    method::M = LBFGS(),
    print_error::Bool = true,
)::T where {T<:Real, M<:Optim.AbstractOptimizer}
    return -minimize_unbounded(x -> -f(x), x0, -y0, optim_iter; method, print_error)
end

function minimize_bounded(
    f::Function, 
    a::Vector{T}, 
    b::Vector{T}, 
    x0::Vector{T}, 
    y0::T, 
    optim_iter::Integer; 
    method::M = LBFGS(),
    print_error::Bool = false,
)::T where {T<:Real, M<:Optim.AbstractOptimizer}
    @assert a < b && optim_iter > 0

    try
        y = minimum(optimize(f, prevfloat.(a), nextfloat.(b), x0, Fminbox(method), 
            Optim.Options(iterations = optim_iter, outer_iterations = optim_iter)
        ))
        return min(y, y0)
    catch e
        if print_error
            println("Optim error: ", e)
        end
        return y0
    end
end

function maximize_bounded(
    f::Function, 
    a::Vector{T}, 
    b::Vector{T}, 
    x0::Vector{T}, 
    y0::T, 
    optim_iter::Integer; 
    method::M = LBFGS(),
    print_error::Bool = false,
)::T where {T<:Real, M<:Optim.AbstractOptimizer}
    return -minimize_bounded(x -> -f(x), a, b, x0, -y0, optim_iter; method, print_error)
end

function generate_reduced_fractions_below_one(pmax::Integer, qmax::Integer)::Vector{Rational}
    @assert pmax > 0 && qmax > 0

    fractions = []
    for p in 1:pmax
        for q in (p+1):qmax
            if gcd(p, q) == 1
                push!(fractions, p // q)
            end
        end
    end

    return fractions
end

function is_gabor_frame(A::Real, B::Real; min_tol::Real = 0, max_tol::Real = Inf)::Bool
    return min_tol < A <= B < max_tol
end

function is_gabor_frame(
    A::Vector{<:Real}, 
    B::Vector{<:Real}; 
    min_tol::Real = 0, 
    max_tol::Real = Inf
)::BitVector
    return is_gabor_frame.(A, B; min_tol, max_tol)
end

function is_not_frame_rationally_oversampled_bspline(
    m::Integer,
)::Function
    @assert m > 0

    return (alpha, beta, p, q) -> beta > 3/2 && abs(beta - round(beta)) * m * q <= 1
end

function save_bounds(
    path::AbstractString, 
    A::Vector{<:Real}, 
    B::Vector{<:Real}, 
    alpha::Vector{<:Real}, 
    beta::Vector{<:Real}
)
    npzwrite(path, Dict("A" => A, "B" => B, "alpha" => alpha, "beta" => beta))
end

function load_bounds(
    path::AbstractString
)::Tuple{Vector{<:Real}, Vector{<:Real}, Vector{<:Real}, Vector{<:Real}}
    d = npzread(path)
    return d["A"], d["B"], d["alpha"], d["beta"]
end

function save_svd_ratio(
    path::AbstractString, 
    svd_ratio::Vector{<:Real}, 
    alpha::Vector{<:Real}, 
    beta::Vector{<:Real}
)
    npzwrite(path, Dict("svd_ratio" => svd_ratio, "alpha" => alpha, "beta" => beta))
end

function load_svd_ratios(
    path::AbstractString
)::Tuple{Vector{<:Real}, Vector{<:Real}, Vector{<:Real}}
    d = npzread(path)
    return d["svd_ratio"], d["alpha"], d["beta"]
end

function save_alpha_beta(
    path::AbstractString, 
    alpha::Vector{<:Real}, 
    beta::Vector{<:Real},
)
    npzwrite(path, Dict("alpha" => alpha, "beta" => beta))
end

function load_alpha_beta(
    path::AbstractString
)::Tuple{Vector{<:Real}, Vector{<:Real}}
    d = npzread(path)
    return d["alpha"], d["beta"]
end

