module Ranking

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Lux: Chain, Dense, WrappedFunction
using Random: AbstractRNG

using LinearAlgebra: dot

"""
$TYPEDEF

Basic benchmark problem with ranking as the CO algorithm.

# Fields
$TYPEDFIELDS
"""
struct RankingBenchmark{W<:AbstractMatrix} <: AbstractStaticBenchmark
    "instances dimension, total number of classes"
    instance_dim::Int
    "number of features"
    nb_features::Int
    "true weight matrix mapping features to costs"
    encoder_weights::W
end

function Base.show(io::IO, bench::RankingBenchmark)
    (; instance_dim, nb_features) = bench
    return print(
        io, "RankingBenchmark(instance_dim=$instance_dim, nb_features=$nb_features)"
    )
end

Utils.objective_value(::RankingBenchmark, sample::DataSample, y) = dot(sample.θ, y)
Utils.is_minimization_problem(::RankingBenchmark) = false

"""
$TYPEDSIGNATURES

Custom constructor for [`RankingBenchmark`](@ref).
"""
function RankingBenchmark(; instance_dim::Int=10, nb_features::Int=5, seed=nothing)
    encoder_weights = randn(Utils.make_rng(seed), Float32, 1, nb_features)
    return RankingBenchmark(instance_dim, nb_features, encoder_weights)
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
    (; instance_dim, nb_features, encoder_weights) = bench
    features = randn(rng, Float32, nb_features, instance_dim)
    θ_true = vec(encoder_weights * features)
    noisy_y_true = ranking(θ_true .+ noise_std * randn(rng, Float32, instance_dim))
    return DataSample(; x=features, θ=θ_true, y=noisy_y_true)
end

"""
$TYPEDSIGNATURES

Returns a Lux model architecture for the ranking benchmark.
"""
function Utils.generate_statistical_model(bench::RankingBenchmark)
    (; nb_features) = bench
    return Chain(Dense(nb_features => 1; use_bias=false), WrappedFunction(vec))
end

export RankingBenchmark

end
