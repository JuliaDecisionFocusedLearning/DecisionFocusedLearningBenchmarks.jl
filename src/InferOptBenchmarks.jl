module InferOptBenchmarks

using Distributions: Uniform, Bernoulli
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Flux: Chain, Dense
using Graphs
using InferOpt
using LinearAlgebra
using Random
using SimpleWeightedGraphs
using SparseArrays

include("interface.jl")

include("Warcraft/Warcraft.jl")

include("shortest_paths.jl")
include("metrics.jl")

export AbstractBenchmark
export generate_dataset, generate_maximizer, generate_statistical_model

export ShortestPathBenchmark
export compute_gap

end # module InferOptBenchmarks
