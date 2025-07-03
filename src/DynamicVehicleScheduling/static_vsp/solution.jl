"""
$TYPEDEF

Solution for the static Vehicle Scheduling Problem.

# Fields
$TYPEDFIELDS
"""
struct VSPSolution
    "list of routes, each route being a list of request indices in corresponding instance (excluding the depot)."
    routes::Vector{Vector{Int}}
    "size (nb_locations, nb_locations). `edge_matrix[i, j]` is equal to 1 if a route takes edge `(i, j)`."
    edge_matrix::BitMatrix
end

"""
$TYPEDSIGNATURES

Get routes from `solution`.
"""
routes(solution::VSPSolution) = solution.routes

"""
$TYPEDSIGNATURES

Get edge matrix from `solution`.
"""
edge_matrix(solution::VSPSolution) = solution.edge_matrix

"""
$TYPEDSIGNATURES

Build a `VSPSolution` from routes. Set `max_index` to manually define the size of the `edge_index` matrix.
"""
function VSPSolution(routes::Vector{Vector{Int}}; max_index=nothing)
    if length(routes) == 0 && isnothing(max_index)
        return VSPSolution(routes, falses(0, 0))
    end
    N = isnothing(max_index) ? maximum(maximum(route) for route in routes) : max_index
    edge_matrix = falses(N, N)
    for route in routes
        old = 1
        for r in route
            edge_matrix[old, r] = true
            old = r
        end
        edge_matrix[old, 1] = true
    end
    return VSPSolution(routes, edge_matrix)
end
