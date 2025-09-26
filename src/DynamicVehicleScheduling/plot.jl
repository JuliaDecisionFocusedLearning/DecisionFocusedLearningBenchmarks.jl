"""
Typed container for plot-ready data.
"""
struct PlotData
    n_epochs::Int
    coordinates::Vector{Vector{Tuple{Float64,Float64}}}
    start_times::Vector{Vector{Float64}}
    node_types::Vector{Vector{Symbol}}
    routes::Vector{Vector{Vector{Int}}}
    epoch_costs::Vector{Float64}
end

function PlotData(d::Dict)
    return PlotData(
        d[:n_epochs],
        d[:coordinates],
        d[:start_times],
        d[:node_types],
        d[:routes],
        d[:epoch_costs],
    )
end

function plot_instance(env::DVSPEnv; kwargs...)
    return plot_instance(env.instance.static_instance; kwargs...)
end

function build_state_data(state::DVSPState)
    coords = coordinate(state)
    x = [p.x for p in coords]
    y = [p.y for p in coords]
    x_depot = x[1]
    y_depot = y[1]
    x_customers = x[2:end]
    y_customers = y[2:end]
    start_times_customers = start_time(state)[2:end]
    service_times_customers = service_time(state)[2:end]
    must_customers = state.is_must_dispatch[2:end]

    return (;
        x_depot=x_depot,
        y_depot=y_depot,
        x_customers=x_customers,
        y_customers=y_customers,
        is_must_dispatch=must_customers,
        start_times=start_times_customers,
        service_times=service_times_customers,
    )
end

"""
$TYPEDSIGNATURES

Plot a given DVSPState showing depot, must-dispatch customers, and postponable customers.
"""
function plot_state(
    state::DVSPState;
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
    (; x_depot, y_depot, x_customers, y_customers, is_must_dispatch, start_times) = build_state_data(
        state
    )

    plot_args = Dict(
        :legend => :topleft, :title => "DVSP State - Epoch $(state.current_epoch)"
    )

    if show_axis_labels
        plot_args[:xlabel] = "x coordinate"
        plot_args[:ylabel] = "y coordinate"
    end

    # Merge with kwargs (possibly overriding defaults)
    for (k, v) in kwargs
        plot_args[k] = v
    end

    fig = plot(; plot_args...)

    # Display depot
    scatter!(
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

    scatter_must_dispatch_args = Dict(
        :label => "Must-dispatch customers",
        :markercolor => must_dispatch_color,
        :marker => must_dispatch_marker,
        :markersize => customer_markersize,
        :markerstrokewidth => markerstrokewidth,
    )

    scatter_postponable_args = Dict(
        :label => "Postponable customers",
        :markercolor => postponable_color,
        :marker => postponable_marker,
        :markersize => customer_markersize,
        :markerstrokewidth => markerstrokewidth,
    )
    if show_colorbar
        scatter_must_dispatch_args[:marker_z] = start_times[is_must_dispatch]
        scatter_postponable_args[:marker_z] = start_times[.!is_must_dispatch]
        # scatter_postponable_args[:label] = "Postponable customers (start time)"
        scatter_postponable_args[:colormap] = :plasma
        scatter_must_dispatch_args[:colormap] = :plasma
        scatter_postponable_args[:colorbar] = :right
        scatter_must_dispatch_args[:colorbar] = :right
        Plots.gr_cbar_width[] = 0.01
    end

    # Display customers, separating must-dispatch and postponable
    if length(x_customers[is_must_dispatch]) > 0
        scatter!(
            fig,
            x_customers[is_must_dispatch],
            y_customers[is_must_dispatch];
            scatter_must_dispatch_args...,
        )
    end

    if length(x_customers[.!is_must_dispatch]) > 0
        scatter!(
            fig,
            x_customers[.!is_must_dispatch],
            y_customers[.!is_must_dispatch];
            scatter_postponable_args...,
        )
    end

    return fig
end

"""
$TYPEDSIGNATURES

Plot a given DVSPState with routes overlaid, showing depot, customers, and vehicle routes.
Routes should be provided as a vector of vectors, where each inner vector contains the
indices of locations visited by that route (excluding the depot).
"""
function plot_routes(
    state::DVSPState,
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
    # Start with the basic state plot
    fig = plot_state(
        state;
        kwargs...,
        title="DVSP State with Routes - Epoch $(state.current_epoch)$cost_text",
    )

    (; x_depot, y_depot, x_customers, y_customers) = build_state_data(state)

    x = vcat(x_depot, x_customers)
    y = vcat(y_depot, y_customers)

    plot_args = Dict(
        :linewidth => route_linewidth, :alpha => route_alpha, :z_order => :back;
    )

    if !isnothing(route_color)
        plot_args[:color] = route_color
    end

    # Plot each route
    for route in routes
        if !isempty(route)
            # Create route path: depot -> customers -> depot
            route_x = vcat(x_depot, x[route], x_depot)
            route_y = vcat(y_depot, y[route], y_depot)

            plot!(fig, route_x, route_y; label=false, plot_args...)
        end
    end

    return fig
end

"""
$TYPEDSIGNATURES

Plot a given DVSPState with routes overlaid. This version accepts routes as a BitMatrix
where entry (i,j) = true indicates an edge from location i to location j.
"""
function plot_routes(state::DVSPState, routes::BitMatrix; kwargs...)
    route_vectors = decode_bitmatrix_to_routes(routes)
    return plot_routes(state, route_vectors; kwargs...)
end

"""
Return a Dict with plot-ready information extracted from a vector of DataSample objects.


The returned dictionary contains:
- :n_epochs => Int
- :coordinates => Vector{Vector{Tuple{Float64,Float64}}} (per-epoch list of (x,y) tuples, empty if instance missing)
- :start_times => Vector{Vector{Float64}} (per-epoch start times, empty if instance missing)
- :node_types => Vector{Vector{Symbol}} (per-epoch node-type labels aligned with coordinates)
- :routes => Vector{Vector{Vector{Int}}} (per-epoch normalized routes; empty vector when no routes)
- :epoch_costs => Vector{Float64} (per-epoch cost; NaN if not computable)

This lets plotting code build figures without depending on plotting internals.
"""
function build_plot_data(data_samples::Vector{<:DataSample})
    return [build_state_data(sample.instance.state) for sample in data_samples]
end

"""
$TYPEDSIGNATURES

Plot multiple epochs side by side from a vector of DataSample objects.
Each DataSample should contain an instance (DVSPState) and optionally y_true (routes).
All subplots will use the same xlims and ylims to show the dynamics clearly.
"""
function plot_epochs(
    data_samples::Vector{<:DataSample};
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
    if length(data_samples) == 0
        error("No data samples provided")
    end

    # Build centralized plot data
    pd = build_plot_data(data_samples)
    n_epochs = length(pd)

    # Determine grid layout
    if isnothing(cols)
        cols = min(n_epochs, 3)  # Default to max 3 columns
    end
    rows = ceil(Int, n_epochs / cols)

    # Calculate global xlims and ylims from all states
    x_min = minimum(min(data.x_depot, minimum(data.x_customers)) for data in pd)
    x_max = maximum(max(data.x_depot, maximum(data.x_customers)) for data in pd)
    y_min = minimum(min(data.y_depot, minimum(data.y_customers)) for data in pd)
    y_max = maximum(max(data.y_depot, maximum(data.y_customers)) for data in pd)

    xlims = (x_min - margin, x_max + margin)
    # Add extra margin at the top for legend space
    y_range = y_max - y_min + 2 * margin
    legend_margin = y_range * legend_margin_factor
    ylims = (y_min - margin, y_max + margin + legend_margin)

    # Calculate global color limits for consistent scaling across subplots
    min_start_time = minimum(minimum(data.start_times) for data in pd)
    max_start_time = maximum(maximum(data.start_times) for data in pd)
    clims = (min_start_time, max_start_time)

    # Create subplots
    plots = map(1:n_epochs) do i
        sample = data_samples[i]
        state = sample.instance.state
        reward = sample.instance.reward

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

        if plot_routes_flag && !isnothing(sample.y_true)
            fig = plot_routes(
                state,
                sample.y_true;
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

    # Calculate dynamic figure size if not specified
    if isnothing(figsize)
        plot_width = 600 * cols
        plot_height = 500 * rows
        figsize = (plot_width, plot_height)
    end

    # Combine plots in a grid layout with optional shared colorbar
    if show_colorbar
        combined_plot = plot(
            plots...; layout=(rows, cols), size=figsize, link=:both, clims=clims
        )
    else
        combined_plot = plot(
            plots...; layout=(rows, cols), size=figsize, link=:both, clims=clims
        )
    end

    return combined_plot
end

"""
$TYPEDSIGNATURES

Plot multiple epochs side by side, optionally filtering to specific epoch indices.
"""
function plot_epochs(
    data_samples::Vector{<:DataSample}, epoch_indices::Vector{Int}; kwargs...
)
    filtered_samples = data_samples[epoch_indices]
    return plot_epochs(filtered_samples; kwargs...)
end

"""
$TYPEDSIGNATURES

Create an animated GIF showing the evolution of states and routes over epochs.
Each frame shows the state and routes for one epoch.
"""
function animate_epochs(
    data_samples::Vector{<:DataSample};
    filename="dvsp_animation.gif",
    fps=1,
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
    if length(data_samples) == 0
        error("No data samples provided")
    end

    pd = build_plot_data(data_samples)
    n_epochs = pd.n_epochs
    # Build all_coordinates from pd.coordinates
    all_coordinates = []
    for coords in pd.coordinates
        for c in coords
            push!(all_coordinates, (; x=c[1], y=c[2]))
        end
    end
    epoch_costs = copy(pd.epoch_costs)

    if isempty(all_coordinates)
        error("No valid coordinates found in data samples")
    end

    # Calculate cumulative costs for the cost bar
    cumulative_costs = cumsum(epoch_costs)
    max_cost = isempty(cumulative_costs) ? 1.0 : maximum(cumulative_costs)

    xlims = (
        minimum(p.x for p in all_coordinates) - margin,
        maximum(p.x for p in all_coordinates) + margin,
    )

    # Add extra margin at the top for legend space and cost bar
    y_min = minimum(p.y for p in all_coordinates) - margin
    y_max = maximum(p.y for p in all_coordinates) + margin
    y_range = y_max - y_min
    legend_margin = y_range * legend_margin_factor

    # Adjust x-axis if showing cost bar
    if show_cost_bar
        x_min, x_max = xlims
        x_range = x_max - x_min
        cost_bar_space = x_range * (cost_bar_width + cost_bar_margin)
        xlims = (x_min, x_max + cost_bar_space)
    end

    ylims = (y_min, y_max + legend_margin)

    # Calculate global color limits
    all_start_times = []
    for sample in data_samples
        if !isnothing(sample.instance.state)
            times = start_time(sample.instance.state)
            append!(all_start_times, times)
        end
    end

    clims = if !isempty(all_start_times)
        (minimum(all_start_times), maximum(all_start_times))
    else
        (0.0, 1.0)
    end

    # Helper function to check if routes exist and are non-empty
    function has_routes(routes)
        if isnothing(routes)
            return false
        elseif routes isa Vector{Vector{Int}}
            return any(!isempty(route) for route in routes)
        elseif routes isa Vector{Int}
            return !isempty(routes)
        elseif routes isa BitMatrix
            return any(routes)
        else
            return false
        end
    end

    # Create frame plan: determine which epochs have routes
    frame_plan = []
    for (epoch_idx, sample) in enumerate(data_samples)
        # Always add state frame
        push!(frame_plan, (epoch_idx, :state))

        # Add routes frame only if routes exist
        if has_routes(sample.y_true)
            push!(frame_plan, (epoch_idx, :routes))
        end
    end

    total_frames = length(frame_plan)

    # Create animation with dynamic frame plan
    anim = @animate for frame_idx in 1:total_frames
        epoch_idx, frame_type = frame_plan[frame_idx]
        sample = data_samples[epoch_idx]
        state = sample.instance.state

        if isnothing(state)
            # Empty frame for missing data
            fig = plot(;
                xlims=xlims,
                ylims=ylims,
                title="Epoch $epoch_idx (No Data)",
                titlefontsize=titlefontsize,
                guidefontsize=guidefontsize,
                tickfontsize=tickfontsize,
                legend=false,
                size=figsize,
                kwargs...,
            )
        else
            if frame_type == :routes
                # Show state with routes
                fig = plot_routes(
                    state,
                    sample.y_true;
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
            else # frame_type == :state
                # Show state only
                fig = plot_state(
                    state;
                    xlims=xlims,
                    ylims=ylims,
                    clims=clims,
                    title="Epoch $(state.current_epoch) - Available Requests",
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
        end

        # Add cost bar if requested
        if show_cost_bar && !isempty(cumulative_costs)
            # Calculate cost bar position on the right side of the plot
            x_min, x_max = xlims
            x_range = x_max - x_min
            bar_x_start = x_max - cost_bar_width * x_range
            bar_x_end = x_max - cost_bar_margin * x_range

            y_min, y_max = ylims
            y_range = y_max - y_min
            bar_y_start = y_min + 0.1 * y_range
            bar_y_end = y_max - 0.1 * y_range
            bar_height = bar_y_end - bar_y_start

            # Calculate current cumulative cost based on frame type
            # Cost increases only when routes are displayed (dispatched)
            current_cost = 0.0

            # Go through all frames up to the current one to see which epochs have had routes dispatched
            for frame_i in 1:frame_idx
                frame_epoch, frame_frame_type = frame_plan[frame_i]

                # Add cost only when we encounter a routes frame
                if frame_frame_type == :routes && frame_epoch <= length(epoch_costs)
                    current_cost += epoch_costs[frame_epoch]
                end
            end

            # Calculate filled height
            if max_cost > 0
                filled_height = (current_cost / max_cost) * bar_height
            else
                filled_height = 0.0
            end

            # Draw the cost bar background (empty bar)
            plot!(
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
            # Draw the filled portion with solid color
            if filled_height > 0
                # Get a color at a value between 0 and 1
                ratio = current_cost / max_cost
                color_at_val = Plots.get(cmap, ratio)
                plot!(
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

            # Add current cost value
            cost_text_y = bar_y_start + filled_height + 0.02 * y_range
            if cost_text_y > bar_y_end
                cost_text_y = bar_y_end
            end

            plot!(
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

            # Add cost bar title
            plot!(
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

    # Save as GIF
    gif(anim, filename; fps=fps)

    return anim
end

"""
Animate multiple solutions where each solution is provided as its own vector of
`DataSample` objects (one per epoch). This treats each solution's `DataSample`
as the canonical source for that column in the side-by-side animation.
"""
function animate_solutions_side_by_side(
    solutions_data_samples::AbstractVector;
    solution_names=nothing,
    filename="dvsp_solutions_side_by_side.gif",
    fps=1,
    figsize=(1200, 600),
    margin=0.1,
    legend_margin_factor=0.15,
    titlefontsize=14,
    guidefontsize=12,
    legendfontsize=11,
    tickfontsize=10,
    show_axis_labels=true,
    show_cost_bar=true,
    cost_bar_width=0.05,
    cost_bar_margin=0.02,
    cost_bar_color_palette=:turbo,
    kwargs...,
)
    n_solutions = length(solutions_data_samples)
    if n_solutions == 0
        error("No solutions provided")
    end

    # Ensure all solution sequences have the same number of epochs
    n_epochs = length(solutions_data_samples[1])
    for (i, s) in enumerate(solutions_data_samples)
        if length(s) != n_epochs
            error(
                "All solution DataSample vectors must have the same length. Solution $i has length $(length(s)) but expected $n_epochs",
            )
        end
    end

    if isnothing(solution_names)
        solution_names = ["Solution $(i)" for i in 1:n_solutions]
    end

    # Collect global coordinates and start times across all solutions/epochs
    all_coordinates = []
    all_start_times = []
    epoch_costs_per_solution = [Float64[] for _ in 1:n_solutions]

    for j in 1:n_solutions
        samples = solutions_data_samples[j]
        for (t, sample) in enumerate(samples)
            if !isnothing(sample.instance.state)
                append!(all_coordinates, coordinate(sample.instance.state))
                append!(all_start_times, start_time(sample.instance.state))

                if sample.y_true isa BitMatrix
                    routes = decode_bitmatrix_to_routes(sample.y_true)
                else
                    routes = sample.y_true isa Vector{Int} ? [sample.y_true] : sample.y_true
                end
                c = isnothing(routes) ? 0.0 : cost(sample.instance.state, routes)
                push!(epoch_costs_per_solution[j], c)
            else
                push!(epoch_costs_per_solution[j], NaN)
            end
        end
    end

    if isempty(all_coordinates)
        error("No valid coordinates found in solution data samples")
    end

    # Global limits
    xlims = (
        minimum(p.x for p in all_coordinates) - margin,
        maximum(p.x for p in all_coordinates) + margin,
    )

    y_min = minimum(p.y for p in all_coordinates) - margin
    y_max = maximum(p.y for p in all_coordinates) + margin
    y_range = y_max - y_min
    legend_margin = y_range * legend_margin_factor
    ylims = (y_min, y_max + legend_margin)

    clims = if !isempty(all_start_times)
        (minimum(all_start_times), maximum(all_start_times))
    else
        (0.0, 1.0)
    end

    # Robust cumulative costs per solution
    robust_cumulative = Vector{Vector{Float64}}(undef, n_solutions)
    for j in 1:n_solutions
        robust = Float64[]
        s = 0.0
        for c in epoch_costs_per_solution[j]
            if !isnan(c)
                s += c
            end
            push!(robust, s)
        end
        robust_cumulative[j] = robust
    end

    function has_routes_local(routes)
        if isnothing(routes)
            return false
        elseif routes isa Vector{Vector{Int}}
            return any(!isempty(route) for route in routes)
        elseif routes isa Vector{Int}
            return !isempty(routes)
        elseif routes isa BitMatrix
            return any(routes)
        else
            return false
        end
    end

    anim = @animate for t in 1:n_epochs
        col_plots = []
        for j in 1:n_solutions
            sample = solutions_data_samples[j][t]
            state = sample.instance.state
            routes = sample.y_true

            if isnothing(state)
                fig = plot(;
                    xlims=xlims,
                    ylims=ylims,
                    title="$(solution_names[j]) - Epoch $t (No Data)",
                    titlefontsize=titlefontsize,
                    guidefontsize=guidefontsize,
                    tickfontsize=tickfontsize,
                    legend=false,
                    kwargs...,
                )
            else
                if has_routes_local(routes)
                    fig = plot_routes(
                        state,
                        routes;
                        xlims=xlims,
                        ylims=ylims,
                        clims=clims,
                        title="$(solution_names[j]) - Epoch $(state.current_epoch)",
                        titlefontsize=titlefontsize,
                        guidefontsize=guidefontsize,
                        legendfontsize=legendfontsize,
                        tickfontsize=tickfontsize,
                        show_axis_labels=show_axis_labels,
                        markerstrokewidth=0.5,
                        show_route_labels=false,
                        show_colorbar=false,
                        size=(floor(Int, figsize[1] / n_solutions), figsize[2]),
                        kwargs...,
                    )
                else
                    fig = plot_state(
                        state;
                        xlims=xlims,
                        ylims=ylims,
                        clims=clims,
                        title="$(solution_names[j]) - Epoch $(state.current_epoch)",
                        titlefontsize=titlefontsize,
                        guidefontsize=guidefontsize,
                        legendfontsize=legendfontsize,
                        tickfontsize=tickfontsize,
                        show_axis_labels=show_axis_labels,
                        markerstrokewidth=0.5,
                        size=(floor(Int, figsize[1] / n_solutions), figsize[2]),
                        kwargs...,
                    )
                end
            end

            # cost bar
            if show_cost_bar
                current_cost = robust_cumulative[j][t]
                max_cost = maximum([robust_cumulative[k][end] for k in 1:n_solutions])

                x_min, x_max = xlims
                x_range = x_max - x_min
                bar_x_start = x_max - cost_bar_width * x_range
                bar_x_end = x_max - cost_bar_margin * x_range

                y_min, y_max = ylims
                y_range = y_max - y_min
                bar_y_start = y_min + 0.1 * y_range
                bar_y_end = y_max - 0.1 * y_range
                bar_height = bar_y_end - bar_y_start

                if max_cost > 0
                    filled_height = (current_cost / max_cost) * bar_height
                else
                    filled_height = 0.0
                end

                plot!(
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

                if filled_height > 0
                    cmap = Plots.cgrad(cost_bar_color_palette)
                    ratio = max_cost > 0 ? current_cost / max_cost : 0.0
                    color_at_val = Plots.get(cmap, ratio)
                    plot!(
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
                plot!(
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
            end

            push!(col_plots, fig)
        end

        combined = plot(
            col_plots...; layout=(1, n_solutions), size=figsize, link=:both, clims=clims
        )
        combined
    end

    gif(anim, filename; fps=fps)
    return anim
end
