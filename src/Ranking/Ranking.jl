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
struct RankingBenchmark{E} <: AbstractBenchmark
    "instances dimension, total number of classes"
    instance_dim::Int
    "number of features"
    nb_features::Int
    "true mapping between features and costs"
    encoder::E
end

function Base.show(io::IO, bench::RankingBenchmark)
    (; instance_dim, nb_features) = bench
    return print(
        io, "RankingBenchmark(instance_dim=$instance_dim, nb_features=$nb_features)"
    )
end

"""
$TYPEDSIGNATURES

Custom constructor for [`RankingBenchmark`](@ref).
"""
function RankingBenchmark(; instance_dim::Int=10, nb_features::Int=5, seed=nothing)
    Random.seed!(seed)
    model = Chain(Dense(nb_features => 1; bias=false), vec)
    return RankingBenchmark(instance_dim, nb_features, model)
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

Generate a labeled sample for the ranking problem.
"""
function Utils.generate_sample(
    bench::RankingBenchmark, rng::AbstractRNG; noise_std::Float32=0.0f0
)
    (; instance_dim, nb_features, encoder) = bench
    features = randn(rng, Float32, nb_features, instance_dim)
    costs = encoder(features)
    noisy_solution = ranking(costs .+ noise_std * randn(rng, Float32, instance_dim))
    return DataSample(; x=features, θ_true=costs, y_true=noisy_solution)
end

"""
$TYPEDSIGNATURES

Initialize a linear model for `bench` using `Flux`.
"""
function Utils.generate_statistical_model(bench::RankingBenchmark; seed=nothing)
    (; nb_features) = bench
    Random.seed!(seed)
    return Chain(Dense(nb_features => 1; bias=false), vec)
end

export RankingBenchmark

end
