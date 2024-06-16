include("../utils.jl")

m_min = 10; m_max = 100; beta_min = 1; beta_max = 100
m_range = m_min:10:m_max
beta_range = beta_min:beta_max

file_path = "../Data/tables/table_m_$m_min-$(m_max)_beta_$beta_min-$beta_max.csv"
data = CSV.read(file_path, CSV.Tables.matrix; header=false, delim = ',')
@assert size(data) == (length(m_range), length(beta_range))

println(size(data))

plt = plot(zscale=:log10)
xlabel!("beta")
ylabel!("peak")
# Makes a plot with beta on the x-axis and the peak values on the y-axis with colors denoting for what m
for i in 1:length(m_range)-6
    m = m_range[i]
    row = data[i, :]
    @assert size(row) == size(beta_range)
    plot!(beta_range, row, label = "m=$m")
end
display(plt)
savefig(plt, "../Figures/julia/peaks_high_m_bspline.png")

beta_range = beta_max-50:beta_max
plt = plot(zscale=:log10)
xlabel!("beta")
ylabel!("peak")
for i in 1:length(m_range)-6
    m = m_range[i]
    row = data[i, end-50:end]
    @assert size(row) == size(beta_range)
    plot!(beta_range, row, label = "m=$m")
end
display(plt)
savefig(plt, "../Figures/julia/zoooom_peaks_high_m_bspline.png")
