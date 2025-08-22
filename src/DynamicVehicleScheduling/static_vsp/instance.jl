"""
$TYPEDEF

Instance data structure for the (deterministic and static) Vehicle Scheduling Problem.

# Fields
$TYPEDFIELDS
"""
@kwdef struct StaticInstance{T}
    "coordinates of the locations. The first one is always the depot."
    coordinate::Vector{Point{T}} = Point{Float64}[]
    "service time at each location"
    service_time::Vector{T} = Float64[]
    "start time at each location"
    start_time::Vector{T} = Float64[]
    "duration matrix between locations"
    duration::Matrix{T} = zeros(Float64, 0, 0)
end

function Base.show(io::IO, instance::StaticInstance)
    N = customer_count(instance)
    return print(io, "VSPInstance with $N customers")
end

"""
$TYPEDSIGNATURES

Return the number of locations in `instance` (customers + depot).
"""
location_count(instance::StaticInstance) = length(instance.coordinate)

"""
$TYPEDSIGNATURES

Return the number of customers in `instance` (excluding the depot).
"""
customer_count(instance::StaticInstance) = location_count(instance) - 1

"""
$TYPEDSIGNATURES

Get the service time vector.
"""
service_time(instance::StaticInstance) = instance.service_time

"""
$TYPEDSIGNATURES

Get the coordinates vector.
"""
coordinate(instance::StaticInstance) = instance.coordinate

"""
$TYPEDSIGNATURES

Get the duration matrix.
"""
duration(instance::StaticInstance) = instance.duration

"""
$TYPEDSIGNATURES

Get the start time vector.
"""
start_time(instance::StaticInstance) = instance.start_time
