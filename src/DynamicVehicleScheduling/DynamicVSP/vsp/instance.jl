"""
$TYPEDEF

Instance data structure for the (deterministic and static) Vehicle Scheduling Problem.

# Fields
$TYPEDFIELDS
"""
@kwdef struct VSPInstance{T}
    "coordinates of the locations. The first one is always the depot."
    coordinate::Vector{Point{T}} = Point{Float64}[]
    "service time at each location"
    service_time::Vector{T} = Float64[]
    "start time at each location"
    start_time::Vector{T} = Float64[]
    "duration matrix between locations"
    duration::Matrix{T} = zeros(Float64, 0, 0)
end

function Base.show(io::IO, instance::VSPInstance)
    N = nb_customers(instance)
    return print(io, "VSPInstance with $N customers")
end

"""
$TYPEDSIGNATURES

Return the number of locations in `instance` (customers + depot).
"""
nb_locations(instance::VSPInstance) = length(instance.coordinate)

"""
$TYPEDSIGNATURES

Return the number of customers in `instance` (excluding the depot).
"""
nb_customers(instance::VSPInstance) = nb_locations(instance) - 1

"""
$TYPEDSIGNATURES

Get the service time vector.
"""
service_time(instance::VSPInstance) = instance.service_time

"""
$TYPEDSIGNATURES

Get the coordinates vector.
"""
coordinate(instance::VSPInstance) = instance.coordinate

"""
$TYPEDSIGNATURES

Get the duration matrix.
"""
duration(instance::VSPInstance) = instance.duration

"""
$TYPEDSIGNATURES

Get the start time vector.
"""
start_time(instance::VSPInstance) = instance.start_time
