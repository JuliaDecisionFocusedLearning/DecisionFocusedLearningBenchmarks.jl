"""
$TYPEDEF
"""
abstract type AbstractInstance end

"""
$TYPEDEF

Instance of the stochastic VSP problem.

# Fields
$TYPEDFIELDS
"""
struct Instance{G<:AbstractGraph,M1<:AbstractMatrix,M2<:AbstractMatrix,F,C} <:
       AbstractInstance
    "associated city"
    city::City
    "graph computed from `city` with the `create_VSP_graph(city::City)` method."
    graph::G
    "features matrix computed from `city`"
    features::Matrix{F}
    ""
    slacks::M1
    ""
    delays::M2
    ""
    vehicle_cost::C
    ""
    delay_cost::C
end

"""
$TYPEDEF

Instance of the stochastic VSP problem.

# Fields
$TYPEDFIELDS
"""
struct CompactInstance{G<:AbstractGraph,M1<:AbstractMatrix,M2<:AbstractMatrix,F,C} <:
       AbstractInstance
    "graph computed from `city` with the `create_VSP_graph(city::City)` method"
    graph::G
    "features matrix computed from `city`"
    features::Matrix{F}
    ""
    slacks::M1
    ""
    delays::M2
    ""
    vehicle_cost::C
    ""
    delay_cost::C
end

"""
$TYPEDSIGNATURES

Constructor for [`Instance`](@ref).
Build an `Instance` from a `City`, by computing its graph, features, slacks and delays.
"""
function Instance(city::City)
    graph = create_VSP_graph(city)
    features = compute_features(city)
    slacks = compute_slacks(city, graph)
    delays = compute_delays(city)
    return Instance(
        city, graph, features, slacks, delays, city.vehicle_cost, city.delay_cost
    )
end

"""
$TYPEDSIGNATURES

Constructor for [`CompactInstance`](@ref).
Build a `CompactInstance` from a `City`, by computing its graph, features, slacks and delays.
"""
function CompactInstance(city::City)
    graph = create_VSP_graph(city)
    features = compute_features(city)
    slacks = compute_slacks(city, graph)
    delays = compute_delays(city)
    return CompactInstance(
        graph, features, slacks, delays, city.vehicle_cost, city.delay_cost
    )
end

"""
$TYPEDSIGNATURES

Returns the number of scenarios in instance.
"""
function get_nb_scenarios(instance::AbstractInstance)
    return size(instance.delays, 2)
end

"""
$TYPEDSIGNATURES

Returns the number of tasks in `instance`.
"""
get_nb_tasks(instance::AbstractInstance) = nv(instance.graph) - 2

"""
$TYPEDSIGNATURES

Returns a random instance created with city_kwargs.
"""
function create_random_instance(; city_kwargs...)
    return Instance(create_random_city(; city_kwargs...))
end

"""
$TYPEDSIGNATURES

Returns a random instance created with city_kwargs.
"""
function create_random_compact_instance(; city_kwargs...)
    return CompactInstance(create_random_city(; city_kwargs...))
end
