"""
$TYPEDSIGNATURES

Sample k random different indices from 2 to N+1.
"""
sample_indices(rng::AbstractRNG, k, N) = randperm(rng, N)[1:k] .+ 1

"""
$TYPEDSIGNATURES

Compute the total cost of a set of routes given a distance matrix, i.e. the sum of the distances between each location in the route.
Note that the first location is implicitly assumed to be the depot, and should not appear in the route.
"""
function cost(routes::Vector{Vector{Int}}, duration::AbstractMatrix)
    total = zero(eltype(duration))
    for route in routes
        current_location = 1
        for r in route
            total += duration[current_location, r]
            current_location = r
        end
        total += duration[current_location, 1]
    end
    return total
end

# """
# $TYPEDSIGNATURES

# Readdir all files in `data` with extension `file_extension`.
# """
# function filtered_readdir(data, file_extension=".txt"; kwargs...)
#     return filter(x -> endswith(x, file_extension), readdir(data; kwargs...))
# end
