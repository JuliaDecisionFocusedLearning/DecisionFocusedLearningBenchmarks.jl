has_visualization(::SubsetSelectionBenchmark) = true

function plot_instance(::SubsetSelectionBenchmark, sample::DataSample; kwargs...)
    x = sample.x  # length n feature vector
    n = length(x)
    return Plots.bar(
        1:n,
        Float64.(x);
        legend=false,
        xlabel="Item",
        ylabel="Feature value",
        title="Features x (observable input)",
        color=:steelblue,
        xticks=1:n,
        kwargs...,
    )
end

function plot_solution(::SubsetSelectionBenchmark, sample::DataSample; kwargs...)
    x = sample.x  # length n feature vector
    θ = sample.θ  # length n hidden values
    y = sample.y  # y[i] = true if item i is selected
    n = length(θ)

    p1 = Plots.bar(
        1:n,
        Float64.(x);
        legend=false,
        ylabel="Feature value",
        title="x (features, observable)",
        color=:steelblue,
        xticks=(1:n, fill("", n)),
    )
    p2 = Plots.bar(
        1:n,
        Float64.(θ);
        legend=false,
        ylabel="Value",
        title="θ (true values)",
        color=:steelblue,
        xticks=(1:n, fill("", n)),
    )
    p3 = Plots.heatmap(
        reshape(Float64.(y), 1, n);
        xlabel="Item",
        ylabel="y",
        title="y (selected items)",
        yticks=false,
        xticks=1:n,
        color=:Greens,
        colorbar=false,
    )

    l = Plots.@layout [a{0.35h}; b{0.35h}; c{0.3h}]
    return Plots.plot(p1, p2, p3; layout=l, size=(600, 480), kwargs...)
end
