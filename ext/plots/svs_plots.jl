import DecisionFocusedLearningBenchmarks.StochasticVehicleScheduling:
    Solution, compute_path_list

has_visualization(::StochasticVehicleSchedulingBenchmark) = true
has_visualization(::ContextualStochasticVehicleSchedulingBenchmark) = true

# ── helpers ────────────────────────────────────────────────────────────────────

function _plot_city(city; colormap=:turbo, task_markersize=7, depot_markersize=9, kwargs...)
    (; tasks, district_width, width) = city
    ticks = 0:district_width:width
    max_time = maximum(t.end_time for t in tasks[2:(end - 1)])
    fig = Plots.plot(;
        xlabel="x",
        ylabel="y",
        gridlinewidth=1,
        gridlinealpha=0.3,
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
    for task in tasks[2:(end - 1)]
        (; start_point, end_point) = task
        Plots.plot!(
            fig,
            [start_point.x, end_point.x],
            [start_point.y, end_point.y];
            color=:gray70,
            linewidth=1,
            label=nothing,
        )
        Plots.scatter!(
            fig,
            [start_point.x],
            [start_point.y];
            markersize=task_markersize,
            marker=:rect,
            marker_z=task.start_time,
            colormap=colormap,
            label=nothing,
        )
        Plots.scatter!(
            fig,
            [end_point.x],
            [end_point.y];
            markersize=task_markersize,
            marker=:rect,
            marker_z=task.end_time,
            colormap=colormap,
            label=nothing,
        )
    end
    Plots.scatter!(
        fig,
        [tasks[1].start_point.x],
        [tasks[1].start_point.y];
        label=nothing,
        marker=:rect,
        markersize=depot_markersize,
        markercolor=:black,
    )
    return fig
end

function _plot_routes(fig, city, path_list; route_linewidth=2, route_alpha=0.7)
    (; tasks) = city
    for path in path_list
        X = Float64[]
        Y = Float64[]
        (; end_point) = tasks[path[1]]
        push!(X, end_point.x)
        push!(Y, end_point.y)
        for task_idx in path[2:end]
            (; start_point, end_point) = tasks[task_idx]
            push!(X, start_point.x)
            push!(Y, start_point.y)
            push!(X, end_point.x)
            push!(Y, end_point.y)
        end
        Plots.plot!(
            fig,
            X,
            Y;
            linewidth=route_linewidth,
            alpha=route_alpha,
            label=false,
            z_order=:back,
        )
    end
    return fig
end

function _annotate_districts(fig, city, district_μ, district_σ; fontsize=6)
    (; districts, district_width) = city
    lin = LinearIndices(districts)
    nx, ny = size(districts)
    for ix in 1:nx, iy in 1:ny
        cx = (ix - 0.5) * district_width
        cy = (iy - 0.5) * district_width
        i = lin[ix, iy]
        label = "μ=$(round(district_μ[i]; digits=2))\nσ=$(round(district_σ[i]; digits=2))"
        Plots.annotate!(fig, cx, cy, Plots.text(label, fontsize, :black, :center))
    end
    return fig
end

function _highlight_district(fig, city, district_idx; color=:red, alpha=0.25)
    (; district_width, width) = city
    nx = width ÷ district_width
    ix = (district_idx - 1) % nx + 1
    iy = (district_idx - 1) ÷ nx + 1
    x0 = (ix - 1) * district_width
    y0 = (iy - 1) * district_width
    Plots.plot!(
        fig,
        Plots.Shape(
            [x0, x0 + district_width, x0 + district_width, x0],
            [y0, y0, y0 + district_width, y0 + district_width],
        );
        fillcolor=color,
        fillalpha=alpha,
        linewidth=0,
        label=nothing,
    )
    return fig
end

# ── interface methods ──────────────────────────────────────────────────────────

function plot_context(::StochasticVehicleSchedulingBenchmark, sample::DataSample; kwargs...)
    @assert hasproperty(sample.instance, :city) "Sample does not contain city information."
    return _plot_city(sample.instance.city; kwargs...)
end

function plot_sample(::StochasticVehicleSchedulingBenchmark, sample::DataSample; kwargs...)
    @assert hasproperty(sample.instance, :city) "Sample does not contain city information."
    city = sample.instance.city
    fig = _plot_city(city; kwargs...)
    solution = Solution(sample.y, sample.instance)
    path_list = compute_path_list(solution)
    _plot_routes(fig, city, path_list)
    return fig
end

function plot_context(
    ::ContextualStochasticVehicleSchedulingBenchmark, sample::DataSample; kwargs...
)
    @assert hasproperty(sample.instance, :city) "Sample does not contain city information."
    city = sample.instance.city
    fig = _plot_city(city; kwargs...)
    _annotate_districts(fig, city, sample.district_μ, sample.district_σ)
    if hasproperty(sample.context, :storm_district)
        _highlight_district(fig, city, sample.storm_district)
    end
    return fig
end

function plot_sample(
    ::ContextualStochasticVehicleSchedulingBenchmark, sample::DataSample; kwargs...
)
    @assert hasproperty(sample.instance, :city) "Sample does not contain city information."
    city = sample.instance.city
    fig = _plot_city(city; kwargs...)
    _annotate_districts(fig, city, sample.district_μ, sample.district_σ)
    if hasproperty(sample.context, :storm_district)
        _highlight_district(fig, city, sample.storm_district)
    end
    if !isnothing(sample.y)
        solution = Solution(sample.y, sample.instance)
        path_list = compute_path_list(solution)
        _plot_routes(fig, city, path_list)
    end
    return fig
end
