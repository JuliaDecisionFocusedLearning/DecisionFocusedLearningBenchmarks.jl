module Warcraft

using ..Utils

using DataDeps: @datadep_str
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Flux
using Graphs
using Images
using LinearAlgebra
using Metalhead
using NPZ
using Random
using SimpleWeightedGraphs
using SparseArrays

include("utils.jl")

"""
$TYPEDEF

Benchmark for the Warcraft shortest path problem.
Does not have any field.
"""
struct WarcraftBenchmark <: AbstractStaticBenchmark end

function Utils.objective_value(::WarcraftBenchmark, sample::DataSample, y::AbstractArray)
    return -dot(sample.θ, y)
end

"""
$TYPEDSIGNATURES

Downloads and decompresses the Warcraft dataset the first time it is called.

!!! warning
    `dataset_size` is capped at 10000, i.e. the number of available samples in the dataset files.
"""
function Utils.generate_dataset(
    ::WarcraftBenchmark,
    dataset_size::Int=10;
    target_policy=nothing,
    seed=nothing,
    rng=MersenneTwister(seed),
    kwargs...,
)
    decompressed_path = datadep"warcraft/data"
    dataset = create_dataset(decompressed_path, dataset_size)
    return isnothing(target_policy) ? dataset : target_policy.(dataset)
end

"""
$TYPEDSIGNATURES

Returns an optimization algorithm that computes a longest path on the grid graph with given weights.
Uses a shortest path algorithm on opposite weights to get the longest path.
"""
function Utils.generate_maximizer(::WarcraftBenchmark; dijkstra=true)
    return dijkstra ? dijkstra_maximizer : bellman_maximizer
end

"""
$TYPEDSIGNATURES

Create and return a `Flux.Chain` embedding for the Warcraft terrains, inspired by [differentiation of blackbox combinatorial solvers](https://github.com/martius-lab/blackbox-differentiation-combinatorial-solvers/blob/master/models.py).

The embedding is made as follows:
1) The first 5 layers of ResNet18 (convolution, batch normalization, relu, maxpooling and first resnet block).
2) An adaptive maxpooling layer to get a (12x12x64) tensor per input image.
3) An average over the third axis (of size 64) to get a (12x12x1) tensor per input image.
4) The element-wize `neg_tensor` function to get cell weights of proper sign to apply shortest path algorithms.
5) A squeeze function to forget the two last dimensions.
"""
function Utils.generate_statistical_model(::WarcraftBenchmark; seed=nothing)
    Random.seed!(seed)
    resnet18 = ResNet(18; pretrain=false, nclasses=1)
    model_embedding = Chain(
        resnet18.layers[1][1][1],
        resnet18.layers[1][1][2],
        resnet18.layers[1][1][3],
        resnet18.layers[1][2][1],
        AdaptiveMaxPool((12, 12)),
        average_tensor,
        neg_tensor,
        squeeze_last_dims,
    )
    return model_embedding
end

export WarcraftBenchmark

end
