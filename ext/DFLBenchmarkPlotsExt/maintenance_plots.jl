has_visualization(::MaintenanceBenchmark) = true

function _degradation_colors(state, n)
    return [s == n ? :firebrick : :steelblue for s in state]
end

function plot_context(bench::MaintenanceBenchmark, sample::DataSample; kwargs...)
    state = sample.instance
    N = length(state)
    n = bench.n
    return Plots.bar(
        1:N,
        state;
        legend=false,
        xlabel="Component",
        ylabel="Degradation level",
        title="Degradation state$(_step_str(sample))",
        ylim=(0, n + 0.5),
        color=_degradation_colors(state, n),
        kwargs...,
    )
end

function plot_sample(bench::MaintenanceBenchmark, sample::DataSample; kwargs...)
    state = sample.instance
    y = sample.y
    N = length(state)
    n = bench.n
    colors = [y[i] ? :seagreen : c for (i, c) in enumerate(_degradation_colors(state, n))]
    return Plots.bar(
        1:N,
        state;
        legend=false,
        xlabel="Component",
        ylabel="Degradation level",
        title="Maintenance$(_step_str(sample))",
        ylim=(0, n + 0.5),
        color=colors,
        kwargs...,
    )
end

function plot_trajectory(
    bench::MaintenanceBenchmark,
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
