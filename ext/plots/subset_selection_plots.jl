has_visualization(::SubsetSelectionBenchmark) = true

function plot_instance(::SubsetSelectionBenchmark, sample::DataSample; kwargs...)
    θ = sample.θ
    n = length(θ)
    return Plots.bar(
        1:n,
        θ;
        legend=false,
        xlabel="Item",
        ylabel="Value",
        title="Instance (values θ)",
        color=:steelblue,
        kwargs...,
    )
end

function plot_solution(::SubsetSelectionBenchmark, sample::DataSample; kwargs...)
    θ = sample.θ
    y = sample.y  # y[i] = true if item i is selected
    n = length(θ)
    colors = [y[i] ? :seagreen : :lightgray for i in 1:n]
    return Plots.bar(
        1:n,
        θ;
        legend=false,
        xlabel="Item",
        ylabel="Value",
        title="Solution (selected items in green)",
        color=colors,
        kwargs...,
    )
end
