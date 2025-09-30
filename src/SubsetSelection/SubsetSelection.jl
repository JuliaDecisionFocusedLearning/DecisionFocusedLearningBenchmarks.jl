module SubsetSelection

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Flux: Chain, Dense
using Random

"""
$TYPEDEF

Benchmark problem for the subset selection problem.
Reference: <https://arxiv.org/abs/2307.13565>.

The goal is to select the best `k` items from a set of `n` items,
without knowing their values, but only observing some features.

# Fields
$TYPEDFIELDS
"""
struct SubsetSelectionBenchmark{M} <: AbstractBenchmark
    "total number of items"
    n::Int
    "number of items to select"
    k::Int
    "hidden unknown mapping from features to costs"
    mapping::M
end

function Base.show(io::IO, bench::SubsetSelectionBenchmark)
    (; n, k) = bench
    return print(io, "SubsetSelectionBenchmark(n=$n, k=$k)")
end

function SubsetSelectionBenchmark(; n::Int=25, k::Int=5, identity_mapping::Bool=true)
    @assert n >= k "number of items n must be greater than k"
    mapping = if identity_mapping
        copy
    else
        Dense(n => n; bias=false)
    end
    return SubsetSelectionBenchmark(n, k, mapping)
end

function top_k(v::AbstractVector, k::Int)
    indices = sortperm(v; rev=true)[1:k]
    res = falses(length(v))
    res[indices] .= true
    return res
end

"""
$TYPEDSIGNATURES

Return a top k maximizer.
"""
function Utils.generate_maximizer(bench::SubsetSelectionBenchmark)
    (; k) = bench
    return Base.Fix2(top_k, k)
end

"""
$TYPEDSIGNATURES

Generate a labeled instance for the subset selection problem.
"""
function Utils.generate_sample(bench::SubsetSelectionBenchmark, rng::AbstractRNG)
    (; n, k, mapping) = bench
    features = randn(rng, Float32, n)
    θ_true = mapping(features)
    y_true = top_k(θ_true, k)
    return DataSample(; x=features, θ=θ_true, y=y_true)
end

"""
$TYPEDSIGNATURES

Initialize a linear model for `bench` using `Flux`.
"""
function Utils.generate_statistical_model(bench::SubsetSelectionBenchmark; seed=0)
    Random.seed!(seed)
    (; n) = bench
    return Dense(n => n; bias=false)
end

export SubsetSelectionBenchmark
export generate_dataset, generate_maximizer, generate_statistical_model

end
