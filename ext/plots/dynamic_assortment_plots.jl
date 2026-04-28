has_visualization(::DynamicAssortmentBenchmark) = true

function plot_instance(::DynamicAssortmentBenchmark, sample::DataSample; kwargs...)
    # sample.instance = (env.features, purchase_history); row 1 of features = prices (×10 to undo normalization)
    prices = sample.instance[1][1, :] .* 10
    N = length(prices)
    return Plots.bar(
        1:N,
        prices;
        legend=false,
        xlabel="Item",
        ylabel="Price",
        title="Instance (item prices): step $(length(sample.instance[2]) + 1)",
        color=:steelblue,
        kwargs...,
    )
end

function plot_solution(::DynamicAssortmentBenchmark, sample::DataSample; kwargs...)
    prices = sample.instance[1][1, :] .* 10
    y = sample.y  # BitVector, selected items
    N = length(prices)
    colors = [y[i] ? :seagreen : :lightgray for i in 1:N]
    return Plots.bar(
        1:N,
        prices;
        legend=false,
        xlabel="Item",
        ylabel="Price",
        title="Assortment (green = offered): step $(length(sample.instance[2]) + 1)",
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
    plots = [plot_solution(bench, trajectory[t]) for t in steps]
    return Plots.plot(
        plots...; layout=(rows, cols), size=(cols * 300, rows * 250), kwargs...
    )
end
