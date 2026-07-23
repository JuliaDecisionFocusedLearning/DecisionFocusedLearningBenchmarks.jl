has_visualization(::DynamicAssortmentBenchmark) = true

function plot_context(::DynamicAssortmentBenchmark, sample::DataSample; kwargs...)
    prices = sample.instance[1][1, :] .* 10
    N = length(prices)
    return Plots.bar(
        1:N,
        prices;
        legend=false,
        xlabel="Item",
        ylabel="Price",
        title="Item prices$(_step_str(sample))",
        color=:steelblue,
        kwargs...,
    )
end

function plot_sample(::DynamicAssortmentBenchmark, sample::DataSample; kwargs...)
    prices = sample.instance[1][1, :] .* 10
    y = sample.y
    N = length(prices)
    colors = [y[i] ? :seagreen : :lightgray for i in 1:N]
    return Plots.bar(
        1:N,
        prices;
        legend=false,
        xlabel="Item",
        ylabel="Price",
        title="Assortment$(_step_str(sample))",
        color=colors,
        kwargs...,
    )
end

function plot_trajectory(
    bench::DynamicAssortmentBenchmark,
    trajectory::Vector{<:DataSample};
    max_steps=6,
    cols=3,
    kwargs...,
)
    n = min(length(trajectory), max_steps)
    rows = ceil(Int, n / cols)
    steps = round.(Int, range(1, length(trajectory); length=n))
    plots = [plot_sample(bench, trajectory[t]) for t in steps]
    return Plots.plot(
        plots...; layout=(rows, cols), size=(cols * 300, rows * 250), kwargs...
    )
end
