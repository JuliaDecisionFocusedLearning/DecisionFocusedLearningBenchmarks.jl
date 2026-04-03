has_visualization(::ArgmaxBenchmark) = true

"""
$TYPEDSIGNATURES

Plot the input features as a heatmap. Columns correspond to items, rows correspond to features.
"""
function plot_instance(::ArgmaxBenchmark, sample::DataSample; kwargs...)
    x = sample.x  # nb_features × n
    n = size(x, 2)
    return Plots.heatmap(
        x;
        xlabel="Item",
        ylabel="Feature",
        title="Features x (observable input)",
        xticks=1:n,
        kwargs...,
    )
end

"""
$TYPEDSIGNATURES

Plot the features `x`, scores `θ`, and decision `y` in `sample` as heatmaps.
All three share the same item axis (columns).
"""
function plot_solution(::ArgmaxBenchmark, sample::DataSample; kwargs...)
    x = sample.x  # nb_features × n
    θ = sample.θ  # length n
    y = sample.y  # one-hot, length n
    n = length(θ)

    p1 = Plots.heatmap(
        x;
        ylabel="Feature",
        title="x (features, observable)",
        xticks=(1:n, fill("", n)),
    )
    θ_min, θ_max = extrema(θ)
    p2 = Plots.heatmap(
        reshape(Float64.(θ), 1, n);
        ylabel="θ",
        title="θ: scores [$(round(θ_min; sigdigits=2)), $(round(θ_max; sigdigits=2))]",
        yticks=false,
        xticks=(1:n, fill("", n)),
        colorbar=false,
    )
    p3 = Plots.heatmap(
        reshape(Float64.(y), 1, n);
        xlabel="Item",
        ylabel="y",
        title="y (decision, one-hot)",
        yticks=false,
        xticks=1:n,
        color=:Greens,
        colorbar=false,
    )

    l = Plots.@layout [a{0.65h}; b{0.175h}; c{0.175h}]
    return Plots.plot(p1, p2, p3; layout=l, size=(600, 420), kwargs...)
end
