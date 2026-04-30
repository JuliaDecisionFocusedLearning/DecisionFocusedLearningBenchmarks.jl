has_visualization(::RankingBenchmark) = true

function plot_instance(::RankingBenchmark, sample::DataSample; kwargs...)
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

function plot_sample(::RankingBenchmark, sample::DataSample; kwargs...)
    x = sample.x  # nb_features × n
    θ = sample.θ  # length n
    y = sample.y  # y[i] = rank of item i (1 = best)
    n = length(θ)

    p1 = Plots.heatmap(
        x; ylabel="Feature", title="x (features, observable)", xticks=(1:n, fill("", n))
    )
    p2 = Plots.bar(
        1:n,
        Float64.(θ);
        legend=false,
        ylabel="Cost",
        title="θ (costs)",
        xticks=(1:n, fill("", n)),
    )
    p3 = Plots.bar(
        1:n,
        Float64.(y);
        legend=false,
        xlabel="Item",
        ylabel="Rank",
        title="y (rank, lower = better)",
        color=:steelblue,
        xticks=1:n,
    )

    l = Plots.@layout [a{0.55h}; b{0.225h}; c{0.225h}]
    return Plots.plot(p1, p2, p3; layout=l, size=(600, 500), kwargs...)
end
