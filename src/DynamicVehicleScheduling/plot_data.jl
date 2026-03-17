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
    state_data = [build_state_data(sample.instance) for sample in data_samples]
    rewards = [sample.reward for sample in data_samples]
    routess = [sample.y for sample in data_samples]
    return [
        (; state..., reward, routes) for
        (state, reward, routes) in zip(state_data, rewards, routess)
    ]
end
