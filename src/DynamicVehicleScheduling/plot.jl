function plot_instancee(env::DVSPEnv; kwargs...)
    return plot_instance(env.instance.static_instance; kwargs...)
end

"""
$TYPEDSIGNATURES

Plot a given DVSPState showing depot, must-dispatch requests, and postponable requests.
"""
function plot_state(
    state::DVSPState;
    customer_markersize=6,
    depot_markersize=8,
    alpha_depot=0.8,
    depot_color=:lightgreen,
    must_dispatch_color=:red,
    postponable_color=:lightblue,
    show_axis_labels=true,
    markerstrokewidth=0.5,
    show_colorbar=false,
    kwargs...,
)
    # Get coordinates from the state instance
    coordinates = coordinate(state)
    start_times = start_time(state)

    # Extract x and y coordinates
    x = [p.x for p in coordinates]
    y = [p.y for p in coordinates]

    # Create the plot
    plot_args = Dict(
        :legend => :topleft, :title => "DVSP State - Epoch $(state.current_epoch)"
    )

    if show_axis_labels
        plot_args[:xlabel] = "x coordinate"
        plot_args[:ylabel] = "y coordinate"
    end

    # Merge with kwargs
    for (k, v) in kwargs
        plot_args[k] = v
    end

    fig = plot(; plot_args...)

    # Plot depot (always the first coordinate)
    scatter!(
        fig,
        [x[1]],
        [y[1]];
        label="Depot",
        markercolor=depot_color,
        marker=:rect,
        markersize=depot_markersize,
        alpha=alpha_depot,
        markerstrokewidth=markerstrokewidth,
    )

    # Plot must-dispatch customers
    if sum(state.is_must_dispatch) > 0
        must_dispatch_indices = findall(state.is_must_dispatch)
        scatter_args = Dict(
            :label => "Must-dispatch requests",
            :markercolor => must_dispatch_color,
            :marker => :star5,
            :markersize => customer_markersize,
            :markerstrokewidth => markerstrokewidth,
        )

        if show_colorbar
            scatter_args[:marker_z] = start_times[must_dispatch_indices]
            scatter_args[:colormap] = :plasma
        end

        scatter!(fig, x[must_dispatch_indices], y[must_dispatch_indices]; scatter_args...)
    end

    # Plot postponable customers
    if sum(state.is_postponable) > 0
        postponable_indices = findall(state.is_postponable)
        scatter_args = Dict(
            :label => "Postponable requests",
            :markercolor => postponable_color,
            :marker => :utriangle,
            :markersize => customer_markersize,
            :markerstrokewidth => markerstrokewidth,
        )

        if show_colorbar
            scatter_args[:marker_z] = start_times[postponable_indices]
            scatter_args[:colormap] = :viridis
        end

        scatter!(fig, x[postponable_indices], y[postponable_indices]; scatter_args...)
    end

    return fig
end

"""
$TYPEDSIGNATURES

Plot a given DVSPState with routes overlaid, showing depot, requests, and vehicle routes.
Routes should be provided as a vector of vectors, where each inner vector contains the
indices of locations visited by that route (excluding the depot).
"""
function plot_routes(
    state::DVSPState,
    routes::Vector{Vector{Int}};
    route_colors=nothing,
    route_linewidth=3,  # Increased from 2 to 3
    route_alpha=0.7,
    show_route_labels=true,
    kwargs...,
)
    # Start with the basic state plot
    fig = plot_state(state; kwargs...)

    # Get coordinates for route plotting
    coordinates = coordinate(state)
    x = [p.x for p in coordinates]
    y = [p.y for p in coordinates]

    # Depot coordinates (always first)
    x_depot = x[1]
    y_depot = y[1]

    # Default route colors if not provided
    if isnothing(route_colors)
        route_colors = [:blue, :purple, :orange, :brown, :pink, :gray, :olive, :cyan]
    end

    # Plot each route
    for (route_idx, route) in enumerate(routes)
        if !isempty(route)
            # Create route path: depot -> customers -> depot
            route_x = vcat(x_depot, x[route], x_depot)
            route_y = vcat(y_depot, y[route], y_depot)

            # Select color for this route
            color = route_colors[(route_idx - 1) % length(route_colors) + 1]

            # Plot the route with more visible styling
            label = show_route_labels ? "Route $route_idx" : nothing
            plot!(
                fig,
                route_x,
                route_y;
                # color=color,
                linewidth=route_linewidth,
                alpha=1.0,  # Make routes fully opaque
                label=label,
                linestyle=:solid,
            )
        end
    end

    return fig
end

"""
$TYPEDSIGNATURES

Plot a given DVSPState with routes overlaid. This version accepts routes as a single
vector where routes are separated by depot visits (index 1).
"""
function plot_routes(state::DVSPState, routes::Vector{Int}; kwargs...)
    # Convert single route vector to vector of route vectors
    route_vectors = Vector{Int}[]
    current_route = Int[]

    for location in routes
        if location == 1  # Depot visit indicates end of route
            if !isempty(current_route)
                push!(route_vectors, copy(current_route))
                empty!(current_route)
            end
        else
            push!(current_route, location)
        end
    end

    # Add the last route if it doesn't end with depot
    if !isempty(current_route)
        push!(route_vectors, current_route)
    end

    return plot_routes(state, route_vectors; kwargs...)
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
$TYPEDSIGNATURES

Plot multiple epochs side by side from a vector of DataSample objects.
Each DataSample should contain an instance (DVSPState) and optionally y_true (routes).
All subplots will use the same xlims and ylims to show the dynamics clearly.
"""
function plot_epochs(
    data_samples::Vector{<:DataSample};
    plot_routes_flag=true,
    cols=nothing,
    figsize=(1800, 600),
    margin=0.05,
    legend_margin_factor=0.15,
    titlefontsize=14,
    guidefontsize=12,
    legendfontsize=11,
    tickfontsize=10,
    show_axis_labels=false,
    show_colorbar=false,
    kwargs...,
)
    n_epochs = length(data_samples)

    if n_epochs == 0
        error("No data samples provided")
    end

    # Determine grid layout
    if isnothing(cols)
        cols = min(n_epochs, 3)  # Default to max 3 columns
    end
    rows = ceil(Int, n_epochs / cols)

    # Calculate global xlims and ylims from all states
    all_coordinates = []
    for sample in data_samples
        if !isnothing(sample.instance)
            coords = coordinate(sample.instance)
            append!(all_coordinates, coords)
        end
    end

    if isempty(all_coordinates)
        error("No valid coordinates found in data samples")
    end

    xlims = (
        minimum(p.x for p in all_coordinates) - margin,
        maximum(p.x for p in all_coordinates) + margin,
    )

    # Add extra margin at the top for legend space
    y_min = minimum(p.y for p in all_coordinates) - margin
    y_max = maximum(p.y for p in all_coordinates) + margin
    y_range = y_max - y_min
    legend_margin = y_range * legend_margin_factor

    ylims = (y_min, y_max + legend_margin)

    # Calculate global color limits for consistent scaling across subplots
    all_start_times = []
    for sample in data_samples
        if !isnothing(sample.instance)
            times = start_time(sample.instance)
            append!(all_start_times, times)
        end
    end

    clims = if !isempty(all_start_times)
        (minimum(all_start_times), maximum(all_start_times))
    else
        (0.0, 1.0)  # Default range
    end

    # Create subplots
    plots = []

    for (i, sample) in enumerate(data_samples)
        state = sample.instance

        if isnothing(state)
            # Create empty plot if no state
            fig = plot(;
                xlims=xlims,
                ylims=ylims,
                title="Epoch $i (No Data)",
                titlefontsize=titlefontsize,
                guidefontsize=guidefontsize,
                tickfontsize=tickfontsize,
                legend=false,
                kwargs...,
            )
        else
            # Plot with or without routes
            if plot_routes_flag && !isnothing(sample.y_true)
                fig = plot_routes(
                    state,
                    sample.y_true;
                    xlims=xlims,
                    ylims=ylims,
                    clims=clims,
                    colorbar=false,
                    title="Epoch $(state.current_epoch)",
                    titlefontsize=titlefontsize,
                    guidefontsize=guidefontsize,
                    legendfontsize=legendfontsize,
                    tickfontsize=tickfontsize,
                    show_axis_labels=show_axis_labels,
                    markerstrokewidth=0.5,
                    show_route_labels=false,
                    kwargs...,
                )
            else
                fig = plot_state(
                    state;
                    xlims=xlims,
                    ylims=ylims,
                    clims=clims,
                    colorbar=false,
                    title="Epoch $(state.current_epoch)",
                    titlefontsize=titlefontsize,
                    guidefontsize=guidefontsize,
                    legendfontsize=legendfontsize,
                    tickfontsize=tickfontsize,
                    show_axis_labels=show_axis_labels,
                    markerstrokewidth=0.5,
                    kwargs...,
                )
            end
        end

        push!(plots, fig)
    end

    # Calculate dynamic figure size if not specified
    if figsize == (1800, 600)  # Using default size
        plot_width = 600 * cols
        plot_height = 500 * rows
        figsize = (plot_width, plot_height)
    end

    # Combine plots in a grid layout with optional shared colorbar
    if show_colorbar
        combined_plot = plot(
            plots...;
            layout=(rows, cols),
            size=figsize,
            link=:both,
            colorbar=:right,
            clims=clims,
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
    kwargs...,
)
    n_epochs = length(data_samples)

    if n_epochs == 0
        error("No data samples provided")
    end

    # Calculate global limits for consistent scaling
    all_coordinates = []
    epoch_costs = Float64[]
    for sample in data_samples
        if !isnothing(sample.instance)
            coords = coordinate(sample.instance)
            append!(all_coordinates, coords)
            # Calculate cost for this epoch if routes exist
            if sample.y_true isa BitMatrix
                routes = decode_bitmatrix_to_routes(sample.y_true)
            else
                routes = sample.y_true isa Vector{Int} ? [sample.y_true] : sample.y_true
            end
            epoch_cost = cost(sample.instance, routes)
            push!(epoch_costs, epoch_cost)
        end
    end

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
        if !isnothing(sample.instance)
            times = start_time(sample.instance)
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
        state = sample.instance

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

            cmap = Plots.cgrad(:turbo)   # or :plasma, :inferno, etc.
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

            # Add cost labels
            # plot!(
            #     fig,
            #     [bar_x_start - 0.01 * x_range],
            #     [bar_y_start];
            #     seriestype=:scatter,
            #     markersize=0,
            #     label="",
            #     annotations=(
            #         bar_x_start - 0.02 * x_range, bar_y_start, ("0", :right, guidefontsize)
            #     ),
            # )

            # if max_cost > 0
            #     plot!(
            #         fig,
            #         [bar_x_start - 0.01 * x_range],
            #         [bar_y_end];
            #         seriestype=:scatter,
            #         markersize=0,
            #         label="",
            #         annotations=(
            #             bar_x_start - 0.02 * x_range,
            #             bar_y_end,
            #             (@sprintf("%.1f", max_cost), :right, guidefontsize),
            #         ),
            #     )
            # end

            # Add current cost value
            cost_text_y = bar_y_start + filled_height + 0.02 * y_range
            if cost_text_y > bar_y_end
                cost_text_y = bar_y_end #+ 0.01 * y_range
            end

            plot!(
                fig,
                [bar_x_start + (bar_x_end - bar_x_start) / 2],
                [cost_text_y];
                seriestype=:scatter,
                markersize=0,
                label="",
                annotations=(
                    bar_x_start - 0.04 * x_range,#(bar_x_start + bar_x_end) / 2,
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
