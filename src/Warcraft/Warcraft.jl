module Warcraft

using ..Utils

using DataDeps: @datadep_str
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Graphs: Graphs, nv, vertices, inneighbors, dijkstra_shortest_paths
using Images: RGB, N0f8
using LinearAlgebra: dot
using Lux:
    Chain,
    Conv,
    InstanceNorm,
    MaxPool,
    AdaptiveMaxPool,
    SkipConnection,
    WrappedFunction,
    relu
using NPZ: npzread
using Random: Random, AbstractRNG
using SimpleWeightedGraphs: SimpleWeightedDiGraph
using SparseArrays: sparse

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
    ::WarcraftBenchmark, dataset_size::Int=10; target_policy=nothing, kwargs...
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

Returns a Lux model architecture for the Warcraft terrain embedding, inspired by
[differentiation of blackbox combinatorial solvers](https://github.com/martius-lab/blackbox-differentiation-combinatorial-solvers/blob/master/models.py).

The architecture consists of:
1) A conv stem (7x7 convolution, instance normalization, relu, max pooling).
2) One residual block with skip connection.
3) An adaptive max pooling layer to get a (12x12x64) tensor per input image.
4) An average over the channel axis to get a (12x12x1) tensor per input image.
5) The element-wise `neg_tensor` function to get cell weights of proper sign.
6) A squeeze function to forget the two last dimensions.

Uses `InstanceNorm` instead of `BatchNorm` since the DFL training loop processes one image at a time.
"""
function Utils.generate_statistical_model(::WarcraftBenchmark)
    return Chain(
        Conv((7, 7), 3 => 64; stride=2, pad=3, use_bias=false),
        InstanceNorm(64, relu),
        MaxPool((3, 3); stride=2, pad=1),
        SkipConnection(
            Chain(
                Conv((3, 3), 64 => 64; pad=1, use_bias=false),
                InstanceNorm(64, relu),
                Conv((3, 3), 64 => 64; pad=1, use_bias=false),
                InstanceNorm(64),
            ),
            +,
        ),
        WrappedFunction(relu),
        AdaptiveMaxPool((12, 12)),
        WrappedFunction(average_tensor),
        WrappedFunction(neg_tensor),
        WrappedFunction(squeeze_last_dims),
    )
end

export WarcraftBenchmark

end
