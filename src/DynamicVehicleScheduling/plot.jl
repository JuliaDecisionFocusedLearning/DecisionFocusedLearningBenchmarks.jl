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

Plot a given DVSPState showing depot, must-dispatch requests, and postponable requests.
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
    show_colorbar=false,
    kwargs...,
)
    (; x_depot, y_depot, x_customers, y_customers, is_must_dispatch, start_times, service_times) = build_state_data(
        state
    )

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
        :label => "Must-dispatch requests",
        :markercolor => must_dispatch_color,
        :marker => must_dispatch_marker,
        :markersize => customer_markersize,
        :markerstrokewidth => markerstrokewidth,
    )

    scatter_postponable_args = Dict(
        :label => "Postponable requests",
        :markercolor => postponable_color,
        :marker => postponable_marker,
        :markersize => customer_markersize,
        :markerstrokewidth => markerstrokewidth,
    )

    scatter_args = Dict()
    if show_colorbar
        scatter_args[:marker_z] = start_times[must_dispatch_indices]
        scatter_args[:colormap] = :plasma
    end

    # Display customers, separating must-dispatch and postponable
    scatter!(
        fig,
        x_customers[is_must_dispatch],
        y_customers[is_must_dispatch];
        scatter_must_dispatch_args...,
        scatter_args...,
    )

    scatter!(
        fig,
        x_customers[.!is_must_dispatch],
        y_customers[.!is_must_dispatch];
        scatter_postponable_args...,
        scatter_args...,
    )

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
    route_color=nothing,
    route_linewidth=2,  # Increased from 2 to 3
    route_alpha=0.8,
    kwargs...,
)
    # Start with the basic state plot
    fig = plot_state(state; kwargs...)

    (; x_depot, y_depot, x_customers, y_customers, is_must_dispatch, start_times, service_times) = build_state_data(
        state
    )

    x = vcat(x_depot, x_customers)
    y = vcat(y_depot, y_customers)

    plot_args = Dict(
        :linewidth => route_linewidth, :alpha => route_alpha, :z_order => :back;
    )

    if !isnothing(route_color)
        plot_args[:color] = route_color
    end

    # Plot each route
    for (route_idx, route) in enumerate(routes)
        if !isempty(route)
            # Create route path: depot -> customers -> depot
            route_x = vcat(x_depot, x[route], x_depot)
            route_y = vcat(y_depot, y[route], y_depot)

            plot!(fig, route_x, route_y; label=false, plot_args...)
        end
    end

    return fig
end

# """
# $TYPEDSIGNATURES

# Plot a given DVSPState with routes overlaid. This version accepts routes as a single
# vector where routes are separated by depot visits (index 1).
# """
# function plot_routes(state::DVSPState, routes::Vector{Int}; kwargs...)
#     # Convert single route vector to vector of route vectors
#     route_vectors = Vector{Int}[]
#     current_route = Int[]

#     for location in routes
#         if location == 1  # Depot visit indicates end of route
#             if !isempty(current_route)
#                 push!(route_vectors, copy(current_route))
#                 empty!(current_route)
#             end
#         else
#             push!(current_route, location)
#         end
#     end

#     # Add the last route if it doesn't end with depot
#     if !isempty(current_route)
#         push!(route_vectors, current_route)
#     end

#     return plot_routes(state, route_vectors; kwargs...)
# end

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
    n = length(data_samples)

    coordinates = Vector{Vector{Tuple{Float64,Float64}}}(undef, n)
    start_times = Vector{Vector{Float64}}(undef, n)
    # node_types[i] is a Vector{Symbol} with one entry per coordinate indicating
    # :depot, :must_dispatch, :postponable, or :customer
    node_types = Vector{Vector{Symbol}}(undef, n)
    routes = Vector{Vector{Vector{Int}}}(undef, n)
    epoch_costs = Float64[]

    for (i, sample) in enumerate(data_samples)
        if isnothing(sample.instance)
            coordinates[i] = Tuple{Float64,Float64}[]
            start_times[i] = Float64[]
            routes[i] = Vector{Vector{Int}}()
            push!(epoch_costs, NaN)
            continue
        end

        # Coordinates and start times
        state = sample.instance
        coords = coordinate(state)
        # convert Point{T} to Tuple{Float64,Float64}
        coordinates[i] = [(p.x, p.y) for p in coords]
        start_times[i] = start_time(state)

        # Build per-node type vector aligned with coords (index 1 == depot)
        types = Symbol[]
        ncoords = length(coords)
        for j in 1:ncoords
            if j == 1
                push!(types, :depot)
            else
                # Guard presence of flags on state; default to :customer
                is_must =
                    hasproperty(state, :is_must_dispatch) &&
                    j <= length(state.is_must_dispatch) &&
                    state.is_must_dispatch[j]
                is_post =
                    hasproperty(state, :is_postponable) &&
                    j <= length(state.is_postponable) &&
                    state.is_postponable[j]
                if is_must
                    push!(types, :must_dispatch)
                elseif is_post
                    push!(types, :postponable)
                else
                    push!(types, :customer)
                end
            end
        end
        node_types[i] = types

        # Normalize routes into Vector{Vector{Int}} (empty vector if no routes)
        r = Vector{Vector{Int}}()
        if !isnothing(sample.y_true)
            if sample.y_true isa BitMatrix
                r = decode_bitmatrix_to_routes(sample.y_true)
            elseif sample.y_true isa Vector{Int}
                # Convert single-vector representation (with depot index 1 separating routes)
                route_vectors = Vector{Vector{Int}}()
                current = Int[]
                for loc in sample.y_true
                    if loc == 1
                        if !isempty(current)
                            push!(route_vectors, copy(current))
                            empty!(current)
                        end
                    else
                        push!(current, loc)
                    end
                end
                if !isempty(current)
                    push!(route_vectors, current)
                end
                r = route_vectors
            elseif sample.y_true isa Vector{Vector{Int}}
                r = sample.y_true
            else
                # Unknown format: try to use as-is, otherwise set nothing
                try
                    # try to coerce to Vector{Vector{Int}}
                    r = Vector{Vector{Int}}(sample.y_true)
                catch
                    r = Vector{Vector{Int}}()
                end
            end
        end

        routes[i] = r

        # Compute cost if possible (keep NaN when no routes)
        if isempty(r)
            push!(epoch_costs, NaN)
        else
            try
                push!(epoch_costs, cost(sample.instance, r))
            catch
                push!(epoch_costs, NaN)
            end
        end
    end

    return PlotData(n, coordinates, start_times, node_types, routes, epoch_costs)
end

"""
Build plot-ready data for multiple solutions. `solutions_data_samples` should be an
AbstractVector where each element is a Vector{<:DataSample} (one per solution). The
function returns a Dict with :n_solutions, :n_epochs, and :solutions => Vector of
the per-solution Dicts produced by `build_plot_data`.
"""
function build_plot_data_for_solutions(solutions_data_samples::AbstractVector)
    n_solutions = length(solutions_data_samples)
    if n_solutions == 0
        return Dict(:n_solutions => 0, :n_epochs => 0, :solutions => Vector{Any}())
    end

    # Ensure consistent epoch length across solutions if possible
    n_epochs = length(solutions_data_samples[1])
    for (i, s) in enumerate(solutions_data_samples)
        if length(s) != n_epochs
            # we won't error here; just record differing lengths in returned dict
            n_epochs = -1
            break
        end
    end

    per_solution = [build_plot_data(s) for s in solutions_data_samples]

    return Dict(
        :n_solutions => n_solutions, :n_epochs => n_epochs, :solutions => per_solution
    )
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
    if length(data_samples) == 0
        error("No data samples provided")
    end

    # Build centralized plot data
    pd = build_plot_data(data_samples)
    n_epochs = pd.n_epochs

    # Determine grid layout
    if isnothing(cols)
        cols = min(n_epochs, 3)  # Default to max 3 columns
    end
    rows = ceil(Int, n_epochs / cols)

    # Calculate global xlims and ylims from all states
    all_coordinates = []
    for coords in pd.coordinates
        for c in coords
            push!(all_coordinates, (; x=c[1], y=c[2]))
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
    for times in pd.start_times
        append!(all_start_times, times)
    end

    clims = if !isempty(all_start_times)
        (minimum(all_start_times), maximum(all_start_times))
    else
        (0.0, 1.0)  # Default range
    end

    # Create subplots
    plots = []

    for i in 1:n_epochs
        sample = data_samples[i]
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
            if !isnothing(sample.instance)
                append!(all_coordinates, coordinate(sample.instance))
                append!(all_start_times, start_time(sample.instance))

                if sample.y_true isa BitMatrix
                    routes = decode_bitmatrix_to_routes(sample.y_true)
                else
                    routes = sample.y_true isa Vector{Int} ? [sample.y_true] : sample.y_true
                end
                c = isnothing(routes) ? 0.0 : cost(sample.instance, routes)
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
            state = sample.instance
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
