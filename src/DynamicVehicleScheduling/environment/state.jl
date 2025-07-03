"""
$TYPEDSIGNATURES

State data structure for the Dynamic Vehicle Scheduling Problem.
"""
@kwdef mutable struct DVSPState{I}
    "current epoch number"
    current_epoch::Int = 1
    "list of location indices from the upper instance (useful for adding new customers)"
    location_indices::Vector{Int} = Int[]
    "associated (static) vehicle scheduling instance"
    state_instance::I = StaticInstance()
    "for each location, 1 if the request must be dispatched, 0 otherwise. The depot is always 0."
    is_must_dispatch::BitVector = falses(0)
    "for each location, 1 if the request can be postponed, 0 otherwise. The depot is always 0."
    is_postponable::BitVector = falses(0)
end

function reset_state!(
    state::DVSPState, instance::Instance; indices, service_time, start_time
)
    (; epoch_duration, Δ_dispatch, static_instance) = instance
    indices_with_depot = vcat(1, indices)
    service_time_with_depot = vcat(0.0, service_time)
    start_time_with_depot = vcat(0.0, start_time)

    coordinates = coordinate(static_instance)[indices_with_depot]
    duration_matrix = duration(static_instance)[indices_with_depot, indices_with_depot]

    is_must_dispatch = falses(length(indices_with_depot))
    is_must_dispatch[2:end] .=
        Δ_dispatch .+ epoch_duration .+ @view(duration_matrix[1, 2:end]) .> start_time

    is_postponable = falses(length(is_must_dispatch))
    is_postponable[2:end] .= .!is_must_dispatch[2:end]

    state.current_epoch = 1
    state.state_instance = StaticInstance(;
        service_time=service_time_with_depot,
        start_time=start_time_with_depot,
        coordinate=coordinates,
        duration=duration_matrix,
    )
    state.is_must_dispatch = is_must_dispatch
    state.is_postponable = is_postponable
    state.location_indices = indices_with_depot
    return nothing
end

function DVSPState(instance::Instance; indices, service_time, start_time)
    state = DVSPState()
    reset_state!(state, instance; indices=indices, service_time=service_time, start_time)
    return state
end

current_epoch(state::DVSPState) = state.current_epoch

"""
$TYPEDSIGNATURES

Return the number of locations in `state` (customers + depot).
"""
location_count(state::DVSPState) = location_count(state.state_instance)

"""
$TYPEDSIGNATURES

Return the number of customers in `state`.
"""
customer_count(state::DVSPState) = customer_count(state.state_instance)

"""
$TYPEDSIGNATURES

Get the service time vector
"""
service_time(state::DVSPState) = service_time(state.state_instance)

"""
$TYPEDSIGNATURES

Get the coordinates vector.
"""
coordinate(state::DVSPState) = coordinate(state.state_instance)

"""
$TYPEDSIGNATURES

Get the duration matrix.
"""
duration(state::DVSPState) = duration(state.state_instance)

"""
$TYPEDSIGNATURES

Get the start time vector.
"""
start_time(state::DVSPState) = start_time(state.state_instance)

"""
$TYPEDSIGNATURES

Check if the given routes are feasible.
Routes should be given with global indexation.
Use [`env_routes_from_state_routes`](@ref) if needed to convert the indices beforehand.
"""
function is_feasible(state::DVSPState, routes::Vector{Vector{Int}}; verbose::Bool=false)
    (; is_must_dispatch, state_instance) = state
    (; duration, start_time, service_time) = state_instance
    is_dispatched = falses(length(is_must_dispatch))

    # Check that routes follow time constraints
    for route in routes
        is_dispatched[route] .= true
        current = 1  # start at the depot
        current_time = start_time[current]
        for next in route
            current_time += duration[current, next]
            if current_time > start_time[next]
                verbose &&
                    @warn "Route $route is infeasible: time constraint violated at location $next"
                return false
            end
            current_time += service_time[next]
            current = next
        end
    end

    # Check that all must dispatch requests are dispatched
    if all(is_dispatched[is_must_dispatch])
        return true
    else
        verbose && @warn "Not all must-dispatch requests are dispatched"
        return false
    end
end

"""
remove dispatched customers, and update must-dispatch and postponable flags.
"""
function apply_routes!(
    state::DVSPState, routes::Vector{Vector{Int}}; check_feasibility::Bool=true
)
    check_feasibility && @assert is_feasible(state, routes; verbose=true)
    (; is_must_dispatch, is_postponable, state_instance, location_indices) = state
    c = cost(state, routes)

    # Remove dispatched customers
    N = location_count(state_instance)
    undispatched_indices = trues(N)
    undispatched_indices[vcat(routes...)] .= false
    state.state_instance = StaticInstance(;
        coordinate=state_instance.coordinate[undispatched_indices],
        service_time=state_instance.service_time[undispatched_indices],
        start_time=state_instance.start_time[undispatched_indices],
        duration=state_instance.duration[undispatched_indices, undispatched_indices],
    )
    state.is_must_dispatch = is_must_dispatch[undispatched_indices]
    state.is_postponable = is_postponable[undispatched_indices]
    state.location_indices = location_indices[undispatched_indices]
    return c
end

function cost(state::DVSPState, routes::Vector{Vector{Int}})
    return cost(routes, duration(state.state_instance))
end

function add_new_customers!(
    state::DVSPState, instance::Instance; indices, service_time, start_time
)
    (; state_instance, is_must_dispatch, is_postponable, location_indices) = state

    updated_indices = vcat(location_indices, indices)
    updated_service_time = vcat(state_instance.service_time, service_time)
    updated_start_time = vcat(state_instance.start_time, start_time)
    updated_coordinates = instance.static_instance.coordinate[updated_indices]
    updated_duration = instance.static_instance.duration[updated_indices, updated_indices]
    is_must_dispatch = falses(length(updated_indices))
    is_postponable = falses(length(updated_indices))

    state.state_instance = StaticInstance(;
        coordinate=updated_coordinates,
        service_time=updated_service_time,
        start_time=updated_start_time,
        duration=updated_duration,
    )

    # Compute must-dispatch flags
    epoch_duration = instance.epoch_duration
    Δ_dispatch = instance.Δ_dispatch
    planning_start_time = (state.current_epoch - 1) * epoch_duration + Δ_dispatch
    is_must_dispatch[2:end] .=
        planning_start_time .+ epoch_duration .+ @view(updated_duration[1, 2:end]) .>
        updated_start_time[2:end]
    is_postponable[2:end] .= .!is_must_dispatch[2:end]

    state.is_must_dispatch = is_must_dispatch
    state.is_postponable = is_postponable
    state.location_indices = updated_indices
    return nothing
end
