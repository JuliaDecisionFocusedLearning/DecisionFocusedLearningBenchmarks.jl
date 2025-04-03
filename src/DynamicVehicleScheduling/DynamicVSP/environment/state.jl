"""
$TYPEDSIGNATURES

State data structure for the Dynamic Vehicle Scheduling Problem.
"""
@kwdef struct VSPState{I}
    "associated (static) vehicle scheduling instance"
    instance::I = VSPInstance()
    "for each location, 1 if the request must be dispatched, 0 otherwise. The depot is always 0."
    is_must_dispatch::BitVector = falses(0)
    "for each location, 1 if the request can be postponed, 0 otherwise. The depot is always 0."
    is_postponable::BitVector = falses(0)
end

"""
$TYPEDSIGNATURES

Return the number of locations in `state` (customers + depot).
"""
nb_locations(state::VSPState) = nb_locations(state.instance)

"""
$TYPEDSIGNATURES

Return the number of customers in `state`.
"""
nb_customers(state::VSPState) = nb_customers(state.instance)

"""
$TYPEDSIGNATURES

Get the service time vector
"""
service_time(state::VSPState) = service_time(state.instance)

"""
$TYPEDSIGNATURES

Get the coordinates vector.
"""
coordinate(state::VSPState) = coordinate(state.instance)

"""
$TYPEDSIGNATURES

Get the duration matrix.
"""
duration(state::VSPState) = duration(state.instance)

"""
$TYPEDSIGNATURES

Get the start time vector.
"""
start_time(state::VSPState) = start_time(state.instance)

"""
$TYPEDSIGNATURES

Check if the given routes are feasible.
Routes should be given with global indexation.
Use [`env_routes_from_state_routes`](@ref) if needed to convert the indices beforehand.
"""
function is_feasible(state::VSPState, routes::Vector{Vector{Int}}; verbose::Bool=false)
    (; is_must_dispatch, instance) = state
    (; duration, start_time, service_time) = instance
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
    return all(is_dispatched[is_must_dispatch])
    return true
end
