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
        scatter!(
            fig,
            x[must_dispatch_indices],
            y[must_dispatch_indices];
            label="Must-dispatch requests",
            markercolor=must_dispatch_color,
            marker=:star5,
            markersize=customer_markersize,
            marker_z=start_times[must_dispatch_indices],
            colormap=:plasma,
            markerstrokewidth=markerstrokewidth,
        )
    end

    # Plot postponable customers
    if sum(state.is_postponable) > 0
        postponable_indices = findall(state.is_postponable)
        scatter!(
            fig,
            x[postponable_indices],
            y[postponable_indices];
            label="Postponable requests",
            markercolor=postponable_color,
            marker=:utriangle,
            markersize=customer_markersize,
            marker_z=start_times[postponable_indices],
            colormap=:viridis,
            markerstrokewidth=markerstrokewidth,
        )
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
    kwargs...,
)
    n_epochs = length(data_samples)

    if n_epochs == 0
        error("No data samples provided")
    end

    # Calculate global limits for consistent scaling
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
            plot(;
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
                plot_routes(
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
                    size=figsize,
                    kwargs...,
                )
            else # frame_type == :state
                # Show state only
                plot_state(
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
                    size=figsize,
                    kwargs...,
                )
            end
        end
    end

    # Save as GIF
    gif(anim, filename; fps=fps)

    return anim
end

# """
# $TYPEDSIGNATURES

# Plot the environment of a DVSPEnv, restricted to the given `epoch_indices` (all epoch if not given).
# """
# function plot_environment(
#     env::DVSPEnv;
#     customer_markersize=4,
#     depot_markersize=7,
#     alpha_depot=0.8,
#     depot_color=:lightgreen,
#     epoch_indices=nothing,
#     kwargs...,
# )
#     draw_all_epochs!(env)

#     epoch_appearance = env.request_epoch
#     coordinates = coordinate(get_state(env))

#     epoch_indices = isnothing(epoch_indices) ? get_epoch_indices(env) : epoch_indices

#     xlims = (minimum(c.x for c in coordinates), maximum(c.x for c in coordinates))
#     ylims = (minimum(c.y for c in coordinates), maximum(c.y for c in coordinates))

#     fig = plot(;
#         legend=:topleft,
#         xlabel="x coordinate",
#         ylabel="y coordinate",
#         xlims,
#         ylims,
#         kwargs...,
#     )

#     for epoch in epoch_indices
#         requests = findall(epoch_appearance .== epoch)
#         x = [coordinates[request].x for request in requests]
#         y = [coordinates[request].y for request in requests]
#         scatter!(
#             fig, x, y; label="Epoch $epoch", marker=:circle, markersize=customer_markersize
#         )
#     end
#     scatter!(
#         fig,
#         [coordinates[1].x],
#         [coordinates[1].y];
#         label="Depot",
#         markercolor=depot_color,
#         marker=:rect,
#         markersize=depot_markersize,
#         alpha=alpha_depot,
#     )

#     return fig
# end

# """
# $TYPEDSIGNATURES

# Plot the given `routes`` for a VSP `state`.
# """
# function plot_epoch(state::DVSPState, routes; kwargs...)
#     (; coordinate, start_time) = state.instance
#     x_depot = coordinate[1].x
#     y_depot = coordinate[1].y
#     X = [p.x for p in coordinate]
#     Y = [p.y for p in coordinate]
#     markersize = 5
#     fig = plot(;
#         legend=:topleft, xlabel="x", ylabel="y", clim=(0.0, maximum(start_time)), kwargs...
#     )
#     for route in routes
#         x_points = vcat(x_depot, X[route], x_depot)
#         y_points = vcat(y_depot, Y[route], y_depot)
#         plot!(fig, x_points, y_points; label=nothing)
#     end
#     scatter!(
#         fig,
#         [x_depot],
#         [y_depot];
#         label="depot",
#         markercolor=:lightgreen,
#         markersize,
#         marker=:rect,
#     )
#     if sum(state.is_postponable) > 0
#         scatter!(
#             fig,
#             X[state.is_postponable],
#             Y[state.is_postponable];
#             label="Postponable customers",
#             marker_z=start_time[state.is_postponable],
#             markersize,
#             colormap=:turbo,
#             marker=:utriangle,
#         )
#     end
#     if sum(state.is_must_dispatch) > 0
#         scatter!(
#             fig,
#             X[state.is_must_dispatch],
#             Y[state.is_must_dispatch];
#             label="Must-dispatch customers",
#             marker_z=start_time[state.is_must_dispatch],
#             markersize,
#             colormap=:turbo,
#             marker=:star5,
#         )
#     end
#     return fig
# end

# """
# $TYPEDSIGNATURES

# Create a plot of routes for each epoch.
# """
# function plot_routes(env::DVSPEnv, routes; epoch_indices=nothing, kwargs...)
#     reset!(env)
#     epoch_indices = isnothing(epoch_indices) ? get_epoch_indices(env) : epoch_indices

#     coordinates = env.config.static_instance.coordinate
#     xlims = (minimum(c.x for c in coordinates), maximum(c.x for c in coordinates))
#     ylims = (minimum(c.y for c in coordinates), maximum(c.y for c in coordinates))

#     figs = map(epoch_indices) do epoch
#         s = next_epoch!(env)
#         fig = plot_epoch(
#             s, state_route_from_env_routes(env, routes[epoch]); xlims, ylims, kwargs...
#         )
#         apply_decision!(env, routes[epoch])
#         return fig
#     end
#     return figs
# end
