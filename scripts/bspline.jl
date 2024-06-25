include("../utils.jl")

dx = 0.01

plt = plot()
x = uniform_spaced_values(-5, 5, dx)
for m in 1:5
    plot!(x, bspline(m, x), label="B-spline $m", xlabel="x", ylabel="y")
end
display(plt)
savefig(plt, "plots/bsplines.png")

plt = plot()
x = uniform_spaced_values(-10, 10, dx)
for m in 1:5
    plot!(x, bspline_hat(m, x), label="Fourier transform of B-spline $m", xlabel="x", ylabel="y")
end
display(plt)
savefig(plt, "plots/bsplines_hat.png")