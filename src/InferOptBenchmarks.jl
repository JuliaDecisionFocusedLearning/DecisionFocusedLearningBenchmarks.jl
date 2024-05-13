module InferOptBenchmarks

using Distributions: Uniform, Bernoulli
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Graphs
using InferOpt
using LinearAlgebra
using Random
using SimpleWeightedGraphs
using SparseArrays

include("Warcraft/Warcraft.jl")

include("interface.jl")
include("shortest_paths.jl")
include("metrics.jl")

export AbstractBenchmark
export ShortestPathBenchmark
export compute_gap

export get_features, get_optimization_parameters, get_solutions, get_maximizer
export input_size, output_size

end # module InferOptBenchmarks
