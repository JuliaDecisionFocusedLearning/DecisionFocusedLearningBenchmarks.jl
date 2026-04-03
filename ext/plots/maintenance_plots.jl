has_visualization(::MaintenanceBenchmark) = true

function plot_instance(bench::MaintenanceBenchmark, sample::DataSample; kwargs...)
    # sample.instance = degradation_state (Vector{Int}, values 1..n)
    state = sample.instance
    N = length(state)
    n = bench.n
    return Plots.bar(
        1:N,
        state;
        legend=false,
        xlabel="Component",
        ylabel="Degradation level",
        title="Instance (degradation state)",
        ylim=(0, n + 0.5),
        color=:steelblue,
        kwargs...,
    )
end

function plot_solution(bench::MaintenanceBenchmark, sample::DataSample; kwargs...)
    state = sample.instance
    y = sample.y  # BitVector, maintained components
    N = length(state)
    n = bench.n
    colors = [y[i] ? :seagreen : (state[i] == n ? :firebrick : :steelblue) for i in 1:N]
    labels = ["comp $i$(y[i] ? " ✓" : "")" for i in 1:N]
    return Plots.bar(
        labels,
        state;
        legend=false,
        ylabel="Degradation level",
        title="Solution (green = maintained, red = failed)",
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
    plots = [plot_solution(bench, trajectory[t]) for t in steps]
    return Plots.plot(
        plots...; layout=(rows, cols), size=(cols * 300, rows * 250), kwargs...
    )
end
