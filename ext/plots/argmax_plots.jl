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

Plot the features `x` as a heatmap, the scores `θ` as a bar chart, and the
decision `y` as a one-hot heatmap. All three share the same item axis.
"""
function plot_sample(::ArgmaxBenchmark, sample::DataSample; kwargs...)
    x = sample.x  # nb_features × n
    θ = sample.θ  # length n
    y = sample.y  # one-hot, length n
    n = length(θ)

    p1 = Plots.heatmap(
        x; ylabel="Feature", title="x (features, observable)", xticks=(1:n, fill("", n))
    )
    p2 = Plots.bar(
        1:n,
        Float64.(θ);
        legend=false,
        ylabel="Score",
        title="θ (scores)",
        xticks=(1:n, fill("", n)),
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

    l = Plots.@layout [a{0.55h}; b{0.3h}; c{0.15h}]
    return Plots.plot(p1, p2, p3; layout=l, size=(600, 480), kwargs...)
end
