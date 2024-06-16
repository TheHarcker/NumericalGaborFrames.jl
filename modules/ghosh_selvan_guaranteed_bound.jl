module GhoshSelvanGuaranteedBound
    module Constants

    const global use_convex_conjecture = false
    const global use_special_cases_of_d = true
    const global assume_sigma_inf_conjecture = false
    end

    module BasicFunctions
    export nu, sigma, sigma_inf, sigma_sup

    using ..Constants 

    function nu(q::T2, m::T2)::T2 where T2<:Integer
        if q == 1 || (!Constants.assume_sigma_inf_conjecture && q <= m / 2)
            return 0
        else
            s = if m >= q
                sum(zeta -> (-1)^zeta * binomial(2 * m, m + q * zeta), 1:floor(Int, m / q))
            else
                0
            end
            return binomial(2 * m, m) + 2 * s
        end
    end

    function sigma_inf(q::Integer, m::Integer)::Real
        return nu(q, m) * q / 4^m
    end

    function sigma_sup(q::Integer, m::Integer)::Real
    @assert m >= 2 && q >= 2
    s = if m >= q 
        sum(zeta -> binomial(2*m, m+q*zeta), 1:floor(Int, m/q)) 
        else 
            0
        end
    value = q / 4^m * (binomial(2*m, m) + 2*s)
    @assert value >= 1
    return value
    end

    function sigma(q::Integer, m::Integer, w::Real)::Real
        return sum(l -> sin(pi * (w+l/q))^(2*m), 1:q)
    end
    
    end
    module NonPeriodicBound
    using ..BasicFunctions
    using ..Constants

    function min(a::Vector{<:Real}, b::Real)::Vector{<:Real}
        c = copy(a)
        c[c .> b] .= b
        return c
    end

    function get_p(q::Integer, beta_max::Real)::Vector{<:Integer}
        return filter(p -> gcd(p,q) == 1, q+1:ceil(Int, beta_max * q))
    end 

    function get_d_from_bound2(m::Integer, omega2_value::Vector{T2}, d_max::Vector{T2}, beta_values::Vector{T2})::Vector{T2} where T2<:Real
        @assert all(size(beta_values) == size(beta_values))
        omega1_2nd_term = beta_values .^ (2-2*m) * (1 + 1/(2*m-3)) + (beta_values - d_max) .^ (2-2*m) .* (1 .+ (beta_values - d_max)./(beta_values * (2*m-3)))
        
        a::Vector{<:Real} = (1 .+ (beta_values + d_max)./(beta_values * (2*m - 3)))
        b::Vector{<:Real} = (1 .+ (beta_values - d_max)./(beta_values * (2*m - 3)))
        omega1_2nd_term = (beta_values + d_max) .^ (2-2*m) .* a + (beta_values - d_max) .^ (2 - 2*m) .* b
        difference = omega2_value - omega1_2nd_term
        @assert all(difference .> 0)

        temp_d::Vector{<:Real} = sqrt.(difference) / pi^m
        # For d values above 1/pi we can freely increset them to d_max. If some values are above d_max we need to decrease them down to d_max
        indices = temp_d .>= min(d_max, 1/pi)
        temp_d[indices] = d_max[indices]
        return temp_d
    end

    function get_d_from_bound2(m::Integer, omega2_value::Vector{T2}, d_max::Real, beta_values::Vector{T2})::Vector{T2} where T2<:Real
        @assert size(omega2_value) == size(beta_values)
        #omega1_2nd_term = beta_values .^ (2-2*m) * (1 + 1/(2*m-3)) + (beta_values - d_max) .^ (2-2*m) .* (1 .+ (beta_values - d_max)./(beta_values * (2*m-3)))
        a::Vector{<:Real} = (1 .+ (beta_values .+ d_max) ./ (beta_values * (2*m - 3)))
        b::Vector{<:Real} = (1 .+ (beta_values .- d_max) ./ (beta_values * (2*m - 3)))
        omega1_2nd_term = (beta_values .+ d_max) .^ (2-2*m) .* a + (beta_values .- d_max) .^ (2 - 2*m) .* b

        difference = omega2_value - omega1_2nd_term
        @assert all(difference .> 0)
        temp_d::Vector{T2} = sqrt.(difference) * (1/pi)^m
        # Increase from 1/pi to d_max
        temp_d[temp_d .> Base.min(T2(d_max), 1/pi)] .= d_max
        return temp_d
    end

    function omega2(p::Vector{<:Integer}, q::Integer, m::Integer, w::Vector{<:Real})::Vector{<:Real}
        @assert size(p) == size(w)
        a::Vector{<:Real} = w .^(2-2*m) .* (1 .+  w ./ ((2*m-3)*p))
        b::Vector{<:Real} = (p/q .- w) .^ (2-2*m) .* (1 .+ (p/q .- w) ./ ((2*m-3)*p))
        return sigma.(q, m, w) .* (a + b) 
    end
    function omega2(p::Vector{<:Integer}, q::Integer, m::Integer, w::Real)::Vector{<:Real}
        a::Vector{<:Real} = w ^(2-2*m) .* (1 .+  w ./ ((2*m-3)*p))
        b::Vector{<:Real} = (p/q .- w) .^ (2-2*m) .* (1 .+ (p/q .- w) ./ ((2*m-3)*p))
        return sigma(q, m, w) .* (a + b) 
    end

    function find_d_slow_good(
        q::Integer, 
        m::Integer,
        p::Vector{<:Integer},
        T2::Type{<:Real} = Float64;
        binary_search_for_d_with_iterations::Integer = 0
        )::Vector{T2}
        @assert all(p .>=2)
        
        best_d::Vector{<:Real} = zeros(size(p))
        
        adjustment::Vector{<:Integer} = 1 .- p.*q .% 2
        p1 = p - adjustment
        omega2_value = omega2(p, q, m,  p1/(2*q))
        temp_d::Vector{T2} = get_d_from_bound2(m, omega2_value, p1/(2*q), p / q)
        best_d = max.(temp_d, best_d)
        if Constants.use_special_cases_of_d
            beta_values = p / q

            if q % 2 == 0
                # w = 1
                indices = p .>= sqrt(7)*q
                omega2_value = omega2(p[indices], q, m, 1)
                temp_d = zeros(size(best_d))
                temp_d[indices] = get_d_from_bound2(m, omega2_value, 1, beta_values[indices])
                best_d = max.(best_d, temp_d)
                # w = 1/q
                @assert all(p .>= 2)
                omega2_value = omega2(p, q, m, 1/q)
                temp_d = get_d_from_bound2(m, omega2_value, 1/q, beta_values)
                best_d = max.(best_d, temp_d)
            else
                # w = 1 - 1/(2q)
                indices = p .>= sqrt(7)*q
                omega2_value = omega2(p[indices], q, m, 1 - 1/(2*q))
                
                temp_d = zeros(size(best_d))
                temp_d[indices] = get_d_from_bound2(m, omega2_value, 1 - 1/(2*q), beta_values[indices])
                best_d = max.(best_d, temp_d)
                # w = 1/(2*q)
                @assert all(p .>= 2)
                omega2_value = omega2(p, q, m, 1/(2*q))
                best_d = max.(best_d, get_d_from_bound2(m, omega2_value, 1/(2*q), beta_values))
            end
        end
    
        if binary_search_for_d_with_iterations > 0
            d_min = best_d
            d_max = -((q % 2) .- p)/(2*q)

            omega2_value = omega2(p, q, m,  p1/(2*q))
            
            iter = 0
            while iter < binary_search_for_d_with_iterations
                iter += 1
                interval_lengths = d_max - d_min
                points_to_test = d_min + interval_lengths / 2
                
                W = Base.min.(points_to_test .^ 2 * pi ^ (2*m), (1/pi) ^ (2 - 2*m))
                omega_1_value = W + beta_values .^ (2-2*m) .* (1+1/(2*m-3)) .+ (beta_values - points_to_test) .^ (2-2*m) .* (1 .+ (beta_values - points_to_test) ./ (beta_values * (2*m-3)))
                valid_d = omega2_value .>= omega_1_value
                invalid_d = omega2_value .< omega_1_value
                d_min[valid_d] = points_to_test[valid_d]
                d_max[invalid_d] = points_to_test[invalid_d]
            end
            best_d = d_min
            end
        return best_d
    end

    function B(m::Integer, w::Vector{<:Real}, p::Vector{<:Integer}, q::Integer; use_omega_1_as_numerator::Bool = false)::Vector{<:Real}
        @assert size(w) == size(p)
        beta_values = p./q
        omega_2_value = w .^ (2-2*m) + (beta_values - w) .^ (2-2*m) + (w .^ (3-2*m) + (beta_values - w) .^ (3-2*m))./((2*m-3)*p)
        omega_1_with_max_d = (pi^(2*m-2) .+ beta_values .^ (2-2*m)) * 2^(2*m-1) / sigma_inf(q, m)

        numerator::Vector{<:Real} = if use_omega_1_as_numerator omega_1_with_max_d else omega_2_value end

        denominator = if Constants.use_convex_conjecture && use_omega_1_as_numerator
            ((w+p-beta_values) .^ (1-2*m) + (p-w) .^ (1-2*m)) ./ ((2*m-1)*p)
        else
            2*(p-beta_values/2).^(1-2*m) ./ ((2*m-1)*p)
        end

        return numerator ./ denominator
    end

    function get_points_non_periodic_bound(
        q::Integer, 
        m::Integer,
        beta_max::Integer,
        T2::Type{<:Real} = Float64;
        binary_search_for_d_with_iterations::Integer = 0,
        use_omega_1_as_numerator::Bool = false
        )::Tuple{Vector{T2}, Vector{T2}}
        
        p = get_p(q, beta_max)
        beta_values = p/q
        if q == 1
            return (zeros(size(beta_values)), beta_values)
        end
        d::Vector{<:Real} = find_d_slow_good(q, m, p; binary_search_for_d_with_iterations)
        at_first_point::Vector{<:Real} = B(m, d, p, q; use_omega_1_as_numerator)
        at_second_point::Vector{<:Real} = B(m, beta_values / 2, p, q; use_omega_1_as_numerator)
        M::Vector{<:Real} = max.(at_first_point, at_second_point)
        max_alpha_values::Vector{<:Real} = 1 ./ (2*sqrt.(M))
        return (max_alpha_values, beta_values)
    end
    end 



    module MarziehBound
        function sinc(w::Vector{T2}, m::Integer)::Vector{T2} where T2<:Real
            return (sin.(pi*w)./(pi*w)) .^ (2*m)
        end

        function get_point_for_beta(beta::T2, m::Integer)::T2 where T2<:Real
            w = collect(LinRange(0, beta / 2, 10000))
            sinc_values = sinc(w, m) + sinc(w .- beta, m)
            infimum::Real = minimum(x->isnan(x) ? Inf : x, sinc_values)
            return pi/2 * sqrt(2*beta / (1+2*m*beta) * infimum)
        end
        function get_points(beta_values::Vector{T2}, m::Integer)::Vector{T2} where T2<:Real
            return get_point_for_beta.(beta_values, m)
        end
    end
end