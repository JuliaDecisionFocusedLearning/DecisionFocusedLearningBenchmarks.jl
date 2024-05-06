module Warcraft

using LinearAlgebra
using SimpleWeightedGraphs
using SparseArrays

include("grid_graph.jl")

export warcraft_grid_graph, index_to_coord, coord_to_index

end
