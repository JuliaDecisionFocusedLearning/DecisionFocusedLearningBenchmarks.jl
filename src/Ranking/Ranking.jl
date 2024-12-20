module Ranking

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Flux: Chain, Dense
using Random

"""
$TYPEDEF

Basic benchmark problem with ranking as the CO algorithm.

# Fields
$TYPEDFIELDS
"""
struct RankingBenchmark <: AbstractBenchmark
    "instances dimension, total number of classes"
    instance_dim::Int
    "number of features"
    nb_features::Int
end

function Base.show(io::IO, bench::RankingBenchmark)
    (; instance_dim, nb_features) = bench
    return print(
        io, "RankingBenchmark(instance_dim=$instance_dim, nb_features=$nb_features)"
    )
end

function RankingBenchmark(; instance_dim::Int=10, nb_features::Int=5)
    return RankingBenchmark(instance_dim, nb_features)
end

"""
$TYPEDSIGNATURES

Compute the vector `r` such that `rᵢ` is the rank of `θᵢ` in `θ`.
"""
function ranking(θ::AbstractVector; rev::Bool=false, kwargs...)
    return invperm(sortperm(θ; rev=rev))
end

"""
$TYPEDSIGNATURES

Return a ranking maximizer.
"""
function Utils.generate_maximizer(bench::RankingBenchmark)
    return ranking
end

"""
$TYPEDSIGNATURES

Generate a dataset of labeled instances for the ranking problem.
"""
function Utils.generate_dataset(
    bench::RankingBenchmark, dataset_size::Int=10; seed::Int=0, noise_std=0.0
)
    (; instance_dim, nb_features) = bench
    rng = MersenneTwister(seed)
    features = [randn(rng, Float32, nb_features, instance_dim) for _ in 1:dataset_size]
    mapping = Chain(Dense(nb_features => 1; bias=false), vec)
    costs = mapping.(features)
    # solutions = ranking.(costs)
    noisy_solutions = [
        ranking(θ .+ noise_std * randn(rng, Float32, instance_dim)) for θ in costs
    ]
    return [
        DataSample(; x, θ_true, y_true) for
        (x, θ_true, y_true) in zip(features, costs, noisy_solutions)
    ]
end

"""
$TYPEDSIGNATURES

Initialize a linear model for `bench` using `Flux`.
"""
function Utils.generate_statistical_model(bench::RankingBenchmark; seed=0)
    Random.seed!(seed)
    (; nb_features) = bench
    return Chain(Dense(nb_features => 1; bias=false), vec)
end

export RankingBenchmark

end
