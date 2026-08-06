import DecisionFocusedLearningBenchmarks.StochasticVehicleScheduling:
    Solution,
    compute_path_list,
    evaluate_scenario,
    get_nb_scenarios,
    build_stochastic_instance

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
    for (route_idx, path) in enumerate(path_list)
        prev_pt = tasks[path[1]].end_point
        for task_idx in path[2:end]
            (; start_point, end_point) = tasks[task_idx]
            for (a, b) in ((prev_pt, start_point), (start_point, end_point))
                mx = (a.x + b.x) / 2
                my = (a.y + b.y) / 2
                Plots.plot!(
                    fig,
                    [a.x, mx],
                    [a.y, my];
                    linewidth=route_linewidth,
                    alpha=route_alpha,
                    label=false,
                    z_order=:back,
                    arrow=true,
                    color=route_idx,
                )
                Plots.plot!(
                    fig,
                    [mx, b.x],
                    [my, b.y];
                    linewidth=route_linewidth,
                    alpha=route_alpha,
                    label=false,
                    z_order=:back,
                    color=route_idx,
                )
            end
            prev_pt = end_point
        end
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

function _plot_stats(solution, instance, nb_vehicles; fontsize=7)
    nb_scenarios = get_nb_scenarios(instance)
    avg_delay = if nb_scenarios == 0
        NaN
    else
        sum(evaluate_scenario(solution, instance, s) for s in 1:nb_scenarios) / nb_scenarios
    end
    vehicle_cost_total = instance.vehicle_cost * nb_vehicles
    delay_cost_total = instance.delay_cost * avg_delay
    total_cost = vehicle_cost_total + delay_cost_total

    text = string(
        "# vehicles: $nb_vehicles    avg delay: $(round(avg_delay; digits=2))\n",
        "vehicle cost: $(round(vehicle_cost_total; digits=1))    ",
        "delay cost: $(round(delay_cost_total; digits=1))\n",
        "total cost: $(round(total_cost; digits=1))",
    )

    p = Plots.plot(;
        framestyle=:none,
        ticks=nothing,
        legend=false,
        grid=false,
        xlims=(0, 1),
        ylims=(0, 1),
        top_margin=(-6)Plots.mm,
        bottom_margin=(-2)Plots.mm,
    )
    Plots.annotate!(p, 0.5, 0.5, Plots.text(text, fontsize, :center, :center))
    return p
end

# ── interface methods ──────────────────────────────────────────────────────────

function plot_context(::StochasticVehicleSchedulingBenchmark, sample::DataSample; kwargs...)
    @assert hasproperty(sample.instance, :city) "Sample does not contain city information."
    return _plot_city(sample.instance.city; kwargs...)
end

function plot_sample(::StochasticVehicleSchedulingBenchmark, sample::DataSample; kwargs...)
    @assert hasproperty(sample.instance, :city) "Sample does not contain city information."
    city = sample.instance.city
    fig = _plot_city(city; kwargs..., bottom_margin=(-20)Plots.mm)
    solution = Solution(sample.y, sample.instance)
    path_list = filter(p -> length(p) > 2, compute_path_list(solution))
    _plot_routes(fig, city, path_list)
    stats = _plot_stats(solution, sample.instance, length(path_list))
    return Plots.plot(fig, stats; layout=Plots.@layout([a; b{0.12h}]), size=(500, 560))
end

function plot_context(
    ::ContextualStochasticVehicleSchedulingBenchmark, sample::DataSample; kwargs...
)
    @assert hasproperty(sample.instance, :city) "Sample does not contain city information."
    city = sample.instance.city
    fig = _plot_city(city; kwargs...)
    _annotate_districts(fig, city, sample.district_μ, sample.district_σ)
    if hasproperty(sample.context, :storm_district)
        _highlight_district(fig, city, sample.storm_district; color=:blue)
    end
    return fig
end

function plot_sample(
    ::ContextualStochasticVehicleSchedulingBenchmark, sample::DataSample; kwargs...
)
    @assert hasproperty(sample.instance, :city) "Sample does not contain city information."
    city = sample.instance.city
    fig = _plot_city(
        city; kwargs..., bottom_margin=(isnothing(sample.y) ? 0 : -20) * Plots.mm
    )
    _annotate_districts(fig, city, sample.district_μ, sample.district_σ)
    if hasproperty(sample.context, :storm_district)
        storm_active = if hasproperty(sample.extra, :scenario)
            sample.extra.scenario.storm_active
        elseif hasproperty(sample.extra, :scenarios)
            any(s -> s.storm_active, sample.extra.scenarios)
        else
            false
        end
        if storm_active
            _highlight_district(fig, city, sample.storm_district)
        else
            _highlight_district(fig, city, sample.storm_district; color=:green)
        end
    end
    if !isnothing(sample.y)
        solution = Solution(sample.y, sample.instance)
        path_list = filter(p -> length(p) > 2, compute_path_list(solution))
        _plot_routes(fig, city, path_list)
        eval_instance = if hasproperty(sample.extra, :scenario)
            build_stochastic_instance(sample.instance, [sample.extra.scenario])
        elseif hasproperty(sample.extra, :scenarios)
            build_stochastic_instance(sample.instance, sample.extra.scenarios)
        else
            sample.instance
        end
        stats = _plot_stats(solution, eval_instance, length(path_list))
        return Plots.plot(fig, stats; layout=Plots.@layout([a; b{0.12h}]), size=(500, 560))
    end
    return fig
end
