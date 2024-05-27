module ShortestPath2

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Distributions
using Flux: Chain, Dense
using Graphs
using LinearAlgebra
using Random
using SparseArrays

include("shortest_paths.jl")

export ShortestPath2Benchmark
export generate_dataset, generate_statistical_model, generate_maximizer
export compute_gap

end
