"""
$TYPEDSIGNATURES

Plot the environment of a DVSPEnv, restricted to the given `epoch_indices` (all epoch if not given).
"""
function plot_environment(
    env::DVSPEnv;
    customer_markersize=4,
    depot_markersize=7,
    alpha_depot=0.8,
    depot_color=:lightgreen,
    epoch_indices=nothing,
    kwargs...,
)
    draw_all_epochs!(env)

    epoch_appearance = env.request_epoch
    coordinates = coordinate(get_state(env))

    epoch_indices = isnothing(epoch_indices) ? get_epoch_indices(env) : epoch_indices

    xlims = (minimum(c.x for c in coordinates), maximum(c.x for c in coordinates))
    ylims = (minimum(c.y for c in coordinates), maximum(c.y for c in coordinates))

    fig = plot(;
        legend=:topleft,
        xlabel="x coordinate",
        ylabel="y coordinate",
        xlims,
        ylims,
        kwargs...,
    )

    for epoch in epoch_indices
        requests = findall(epoch_appearance .== epoch)
        x = [coordinates[request].x for request in requests]
        y = [coordinates[request].y for request in requests]
        scatter!(
            fig, x, y; label="Epoch $epoch", marker=:circle, markersize=customer_markersize
        )
    end
    scatter!(
        fig,
        [coordinates[1].x],
        [coordinates[1].y];
        label="Depot",
        markercolor=depot_color,
        marker=:rect,
        markersize=depot_markersize,
        alpha=alpha_depot,
    )

    return fig
end

"""
$TYPEDSIGNATURES

Plot the given `routes`` for a VSP `state`.
"""
function plot_epoch(state::DVSPState, routes; kwargs...)
    (; coordinate, start_time) = state.instance
    x_depot = coordinate[1].x
    y_depot = coordinate[1].y
    X = [p.x for p in coordinate]
    Y = [p.y for p in coordinate]
    markersize = 5
    fig = plot(;
        legend=:topleft, xlabel="x", ylabel="y", clim=(0.0, maximum(start_time)), kwargs...
    )
    for route in routes
        x_points = vcat(x_depot, X[route], x_depot)
        y_points = vcat(y_depot, Y[route], y_depot)
        plot!(fig, x_points, y_points; label=nothing)
    end
    scatter!(
        fig,
        [x_depot],
        [y_depot];
        label="depot",
        markercolor=:lightgreen,
        markersize,
        marker=:rect,
    )
    if sum(state.is_postponable) > 0
        scatter!(
            fig,
            X[state.is_postponable],
            Y[state.is_postponable];
            label="Postponable customers",
            marker_z=start_time[state.is_postponable],
            markersize,
            colormap=:turbo,
            marker=:utriangle,
        )
    end
    if sum(state.is_must_dispatch) > 0
        scatter!(
            fig,
            X[state.is_must_dispatch],
            Y[state.is_must_dispatch];
            label="Must-dispatch customers",
            marker_z=start_time[state.is_must_dispatch],
            markersize,
            colormap=:turbo,
            marker=:star5,
        )
    end
    return fig
end

"""
$TYPEDSIGNATURES

Create a plot of routes for each epoch.
"""
function plot_routes(env::DVSPEnv, routes; epoch_indices=nothing, kwargs...)
    reset!(env)
    epoch_indices = isnothing(epoch_indices) ? get_epoch_indices(env) : epoch_indices

    coordinates = env.config.static_instance.coordinate
    xlims = (minimum(c.x for c in coordinates), maximum(c.x for c in coordinates))
    ylims = (minimum(c.y for c in coordinates), maximum(c.y for c in coordinates))

    figs = map(epoch_indices) do epoch
        s = next_epoch!(env)
        fig = plot_epoch(
            s, state_route_from_env_routes(env, routes[epoch]); xlims, ylims, kwargs...
        )
        apply_decision!(env, routes[epoch])
        return fig
    end
    return figs
end
