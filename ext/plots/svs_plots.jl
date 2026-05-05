import DecisionFocusedLearningBenchmarks.StochasticVehicleScheduling:
    Solution, compute_path_list

has_visualization(::StochasticVehicleSchedulingBenchmark) = true

function plot_context(::StochasticVehicleSchedulingBenchmark, sample::DataSample; kwargs...)
    @assert hasproperty(sample.instance, :city) "Sample does not contain city information."
    (; tasks, district_width, width) = sample.instance.city
    ticks = 0:district_width:width
    max_time = maximum(t.end_time for t in sample.instance.city.tasks[1:(end - 1)])
    fig = Plots.plot(;
        xlabel="x",
        ylabel="y",
        gridlinewidth=3,
        aspect_ratio=:equal,
        size=(500, 500),
        xticks=ticks,
        yticks=ticks,
        xlims=(-1, width + 1),
        ylims=(-1, width + 1),
        clim=(0.0, max_time),
        label=nothing,
        colorbar_title="Time",
        kwargs...,
    )
    Plots.scatter!(
        fig,
        [tasks[1].start_point.x],
        [tasks[1].start_point.y];
        label=nothing,
        marker=:rect,
        markersize=10,
    )
    Plots.annotate!(
        fig, (tasks[1].start_point.x, tasks[1].start_point.y, Plots.text("0", 10))
    )
    for (i_task, task) in enumerate(tasks[2:(end - 1)])
        (; start_point, end_point) = task
        points = [(start_point.x, start_point.y), (end_point.x, end_point.y)]
        Plots.plot!(fig, points; color=:black, label=nothing)
        Plots.scatter!(
            fig,
            points[1];
            markersize=10,
            marker=:rect,
            marker_z=task.start_time,
            colormap=:turbo,
            label=nothing,
        )
        Plots.scatter!(
            fig,
            points[2];
            markersize=10,
            marker=:rect,
            marker_z=task.end_time,
            colormap=:turbo,
            label=nothing,
        )
        Plots.annotate!(fig, (points[1]..., Plots.text("$(i_task)", 10)))
    end
    return fig
end

function plot_sample(::StochasticVehicleSchedulingBenchmark, sample::DataSample; kwargs...)
    @assert hasproperty(sample.instance, :city) "Sample does not contain city information."
    (; tasks, district_width, width) = sample.instance.city
    ticks = 0:district_width:width
    solution = Solution(sample.y, sample.instance)
    path_list = compute_path_list(solution)
    fig = Plots.plot(;
        xlabel="x",
        ylabel="y",
        legend=false,
        gridlinewidth=3,
        aspect_ratio=:equal,
        size=(500, 500),
        xticks=ticks,
        yticks=ticks,
        xlims=(-1, width + 1),
        ylims=(-1, width + 1),
        kwargs...,
    )
    for path in path_list
        X = Float64[]
        Y = Float64[]
        (; start_point, end_point) = tasks[path[1]]
        (; x, y) = end_point
        push!(X, x)
        push!(Y, y)
        for task in path[2:end]
            (; start_point, end_point) = tasks[task]
            push!(X, start_point.x)
            push!(Y, start_point.y)
            push!(X, end_point.x)
            push!(Y, end_point.y)
        end
        Plots.plot!(fig, X, Y; marker=:circle)
    end
    return fig
end
