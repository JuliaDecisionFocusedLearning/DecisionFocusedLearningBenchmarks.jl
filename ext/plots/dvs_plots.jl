import DecisionFocusedLearningBenchmarks.DynamicVehicleScheduling as DVS
using Printf: @sprintf

has_visualization(::DynamicVehicleSchedulingBenchmark) = true

# ── helpers ────────────────────────────────────────────────────────────────────

function _compute_bounds(pd; margin=0.05, legend_margin_factor=0.15)
    x_min = minimum(min(data.x_depot, minimum(data.x_customers)) for data in pd)
    x_max = maximum(max(data.x_depot, maximum(data.x_customers)) for data in pd)
    y_min = minimum(min(data.y_depot, minimum(data.y_customers)) for data in pd)
    y_max = maximum(max(data.y_depot, maximum(data.y_customers)) for data in pd)

    xlims = (x_min - margin, x_max + margin)
    y_range = y_max - y_min + 2 * margin
    legend_margin = y_range * legend_margin_factor
    ylims = (y_min - margin, y_max + margin + legend_margin)

    min_start_time = minimum(minimum(data.start_times) for data in pd)
    max_start_time = maximum(maximum(data.start_times) for data in pd)
    clims = (min_start_time, max_start_time)

    return (; xlims, ylims, clims)
end

# ── plot_state ───────────────────────────────────────────────────────────────

"""
$TYPEDSIGNATURES

Plot a given DVSPState showing depot, must-dispatch customers, and postponable customers.
"""
function plot_state(
    state::DVS.DVSPState;
    customer_markersize=6,
    depot_markersize=8,
    alpha_depot=0.8,
    depot_color=:lightgreen,
    depot_marker=:rect,
    must_dispatch_color=:red,
    postponable_color=:lightblue,
    must_dispatch_marker=:star5,
    postponable_marker=:utriangle,
    show_axis_labels=true,
    markerstrokewidth=0.5,
    show_colorbar=true,
    kwargs...,
)
    (; x_depot, y_depot, x_customers, y_customers, is_must_dispatch, start_times) =
        DVS.build_state_data(state)

    xlabel = show_axis_labels ? "x coordinate" : ""
    ylabel = show_axis_labels ? "y coordinate" : ""
    fig = Plots.plot(;
        legend=:topleft,
        title="DVSP State - Epoch $(state.current_epoch)",
        xlabel=xlabel,
        ylabel=ylabel,
        kwargs...,
    )

    Plots.scatter!(
        fig,
        [x_depot],
        [y_depot];
        label="Depot",
        markercolor=depot_color,
        marker=depot_marker,
        markersize=depot_markersize,
        alpha=alpha_depot,
        markerstrokewidth=markerstrokewidth,
    )

    colorbar_args = if show_colorbar
        (; colormap=:plasma, colorbar=:right)
    else
        (;)
    end

    if any(is_must_dispatch)
        Plots.scatter!(
            fig,
            x_customers[is_must_dispatch],
            y_customers[is_must_dispatch];
            label="Must-dispatch",
            markercolor=must_dispatch_color,
            marker=must_dispatch_marker,
            markersize=customer_markersize,
            markerstrokewidth=markerstrokewidth,
            marker_z=show_colorbar ? start_times[is_must_dispatch] : nothing,
            colorbar_args...,
        )
    end

    if any(.!is_must_dispatch)
        Plots.scatter!(
            fig,
            x_customers[.!is_must_dispatch],
            y_customers[.!is_must_dispatch];
            label="Postponable",
            markercolor=postponable_color,
            marker=postponable_marker,
            markersize=customer_markersize,
            markerstrokewidth=markerstrokewidth,
            marker_z=show_colorbar ? start_times[.!is_must_dispatch] : nothing,
            colorbar_args...,
        )
    end

    return fig
end

# ── plot_routes ──────────────────────────────────────────────────────────────

function plot_routes(
    state::DVS.DVSPState,
    routes::Vector{Vector{Int}};
    reward=nothing,
    route_color=nothing,
    route_linewidth=2,
    route_alpha=0.8,
    kwargs...,
)
    cost_text = if !isnothing(reward)
        " (" * @sprintf("%.2f", -reward) * ")"
    else
        ""
    end
    fig = plot_state(
        state;
        kwargs...,
        title="DVSP State with Routes - Epoch $(state.current_epoch)$cost_text",
    )

    (; x_depot, y_depot, x_customers, y_customers) = DVS.build_state_data(state)

    x = vcat(x_depot, x_customers)
    y = vcat(y_depot, y_customers)

    plot_args = Dict(
        :linewidth => route_linewidth, :alpha => route_alpha, :z_order => :back;
    )

    if !isnothing(route_color)
        plot_args[:color] = route_color
    end

    for route in routes
        if !isempty(route)
            route_x = vcat(x_depot, x[route], x_depot)
            route_y = vcat(y_depot, y[route], y_depot)
            Plots.plot!(fig, route_x, route_y; label=false, plot_args...)
        end
    end

    return fig
end

function plot_routes(state::DVS.DVSPState, routes::BitMatrix; kwargs...)
    route_vectors = DVS.decode_bitmatrix_to_routes(routes)
    return plot_routes(state, route_vectors; kwargs...)
end

# ── interface methods ────────────────────────────────────────────────────────

function plot_context(
    bench::DynamicVehicleSchedulingBenchmark, sample::DataSample; kwargs...
)
    return plot_state(sample.instance; kwargs...)
end

function plot_sample(
    bench::DynamicVehicleSchedulingBenchmark, sample::DataSample; kwargs...
)
    return plot_routes(sample.instance, sample.y; reward=sample.reward, kwargs...)
end

function plot_trajectory(
    bench::DynamicVehicleSchedulingBenchmark,
    traj::Vector{<:DataSample};
    plot_routes_flag=true,
    cols=nothing,
    figsize=nothing,
    margin=0.05,
    legend_margin_factor=0.15,
    titlefontsize=14,
    guidefontsize=12,
    legendfontsize=11,
    tickfontsize=10,
    show_axis_labels=false,
    show_colorbar=true,
    kwargs...,
)
    if length(traj) == 0
        error("No data samples provided")
    end

    pd = DVS.build_plot_data(traj)
    n_epochs = length(pd)

    if isnothing(cols)
        cols = min(n_epochs, 3)
    end
    rows = ceil(Int, n_epochs / cols)

    (; xlims, ylims, clims) = _compute_bounds(pd; margin, legend_margin_factor)

    plots = map(1:n_epochs) do i
        sample = traj[i]
        state = sample.instance
        reward = sample.reward

        common_kwargs = Dict(
            :xlims => xlims,
            :ylims => ylims,
            :clims => clims,
            :show_colorbar => show_colorbar,
            :titlefontsize => titlefontsize,
            :guidefontsize => guidefontsize,
            :legendfontsize => legendfontsize,
            :tickfontsize => tickfontsize,
            :show_axis_labels => show_axis_labels,
            :markerstrokewidth => 0.5,
        )

        if plot_routes_flag
            fig = plot_routes(
                state,
                sample.y;
                reward=reward,
                show_route_labels=false,
                common_kwargs...,
                kwargs...,
            )
        else
            fig = plot_state(state; common_kwargs..., kwargs...)
        end

        return fig
    end

    if isnothing(figsize)
        plot_width = 600 * cols
        plot_height = 500 * rows
        figsize = (plot_width, plot_height)
    end

    combined_plot = Plots.plot(
        plots...; layout=(rows, cols), size=figsize, link=:both, clims=clims
    )

    return combined_plot
end

function animate_trajectory(
    bench::DynamicVehicleSchedulingBenchmark,
    traj::Vector{<:DataSample};
    figsize=(800, 600),
    margin=0.1,
    legend_margin_factor=0.2,
    titlefontsize=16,
    guidefontsize=14,
    legendfontsize=12,
    tickfontsize=11,
    show_axis_labels=true,
    show_cost_bar=true,
    show_colorbar=false,
    cost_bar_width=0.05,
    cost_bar_margin=0.02,
    cost_bar_color_palette=:turbo,
    kwargs...,
)
    pd = DVS.build_plot_data(traj)
    epoch_costs = [-sample.reward for sample in traj]

    (; xlims, ylims, clims) = _compute_bounds(pd; margin, legend_margin_factor)

    if show_cost_bar
        x_min, x_max = xlims
        x_range = x_max - x_min
        cost_bar_space = x_range * (cost_bar_width + cost_bar_margin)
        xlims = (x_min, x_max + cost_bar_space)
    end

    frame_plan = []
    for (epoch_idx, _) in enumerate(traj)
        push!(frame_plan, (epoch_idx, :state))
        push!(frame_plan, (epoch_idx, :routes))
    end

    total_frames = length(frame_plan)

    anim = @animate for frame_idx in 1:total_frames
        epoch_idx, frame_type = frame_plan[frame_idx]
        sample = traj[epoch_idx]
        state = sample.instance

        if frame_type == :routes
            fig = plot_routes(
                state,
                sample.y;
                xlims=xlims,
                ylims=ylims,
                clims=clims,
                title="Epoch $(state.current_epoch) - Routes Dispatched",
                titlefontsize=titlefontsize,
                guidefontsize=guidefontsize,
                legendfontsize=legendfontsize,
                tickfontsize=tickfontsize,
                show_axis_labels=show_axis_labels,
                markerstrokewidth=0.5,
                show_route_labels=false,
                show_colorbar=show_colorbar,
                size=figsize,
                kwargs...,
            )
        else
            fig = plot_state(
                state;
                xlims=xlims,
                ylims=ylims,
                clims=clims,
                title="Epoch $(state.current_epoch) - Available Customers",
                titlefontsize=titlefontsize,
                guidefontsize=guidefontsize,
                legendfontsize=legendfontsize,
                tickfontsize=tickfontsize,
                show_axis_labels=show_axis_labels,
                markerstrokewidth=0.5,
                show_colorbar=show_colorbar,
                size=figsize,
                kwargs...,
            )
        end

        if show_cost_bar
            x_min, x_max = xlims
            x_range = x_max - x_min
            bar_x_start = x_max - cost_bar_width * x_range
            bar_x_end = x_max - cost_bar_margin * x_range

            y_min, y_max = ylims
            y_range = y_max - y_min
            bar_y_start = y_min + 0.1 * y_range
            bar_y_end = y_max - 0.1 * y_range
            bar_height = bar_y_end - bar_y_start

            current_cost = 0.0
            for frame_i in 1:frame_idx
                frame_epoch, frame_frame_type = frame_plan[frame_i]
                if frame_frame_type == :routes && frame_epoch <= length(epoch_costs)
                    current_cost += epoch_costs[frame_epoch]
                end
            end

            max_cost = sum(epoch_costs)
            if max_cost > 0
                filled_height = (current_cost / max_cost) * bar_height
            else
                filled_height = 0.0
            end

            Plots.plot!(
                fig,
                [bar_x_start, bar_x_end, bar_x_end, bar_x_start, bar_x_start],
                [bar_y_start, bar_y_start, bar_y_end, bar_y_end, bar_y_start];
                seriestype=:shape,
                color=:white,
                alpha=0.8,
                linecolor=:black,
                linewidth=2,
                label="",
            )

            cmap = Plots.cgrad(cost_bar_color_palette)
            if filled_height > 0
                ratio = current_cost / max_cost
                color_at_val = Plots.get(cmap, ratio)
                Plots.plot!(
                    fig,
                    [bar_x_start, bar_x_end, bar_x_end, bar_x_start, bar_x_start],
                    [
                        bar_y_start,
                        bar_y_start,
                        bar_y_start + filled_height,
                        bar_y_start + filled_height,
                        bar_y_start,
                    ];
                    seriestype=:shape,
                    color=color_at_val,
                    alpha=0.7,
                    linecolor=:darkred,
                    linewidth=1,
                    label="",
                )
            end

            cost_text_y = bar_y_start + filled_height + 0.02 * y_range
            if cost_text_y > bar_y_end
                cost_text_y = bar_y_end
            end

            Plots.plot!(
                fig,
                [bar_x_start + (bar_x_end - bar_x_start) / 2],
                [cost_text_y];
                seriestype=:scatter,
                markersize=0,
                label="",
                annotations=(
                    bar_x_start - 0.04 * x_range,
                    cost_text_y,
                    (@sprintf("%.1f", current_cost), :center, guidefontsize),
                ),
            )

            Plots.plot!(
                fig,
                [(bar_x_start + bar_x_end) / 2],
                [bar_y_end + 0.05 * y_range];
                seriestype=:scatter,
                markersize=0,
                label="",
                annotations=(
                    (bar_x_start + bar_x_end) / 2,
                    bar_y_end + 0.05 * y_range,
                    ("Cost", :center, guidefontsize),
                ),
            )
        end

        fig
    end

    return anim
end
