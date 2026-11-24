module Argmax

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Lux: Chain, Dense, WrappedFunction
using Random: AbstractRNG

using LinearAlgebra: dot

"""
$TYPEDEF

Basic benchmark problem with an argmax as the CO algorithm.

# Fields
$TYPEDFIELDS
"""
struct ArgmaxBenchmark{W<:AbstractMatrix} <: AbstractStaticBenchmark
    "instances dimension, total number of classes"
    instance_dim::Int
    "number of features"
    nb_features::Int
    "true weight matrix mapping features to costs"
    encoder_weights::W
end

function Base.show(io::IO, bench::ArgmaxBenchmark)
    (; instance_dim, nb_features) = bench
    return print(
        io, "ArgmaxBenchmark(instance_dim=$instance_dim, nb_features=$nb_features)"
    )
end

Utils.objective_value(::ArgmaxBenchmark, sample::DataSample, y) = dot(sample.θ, y)

"""
$TYPEDSIGNATURES

Custom constructor for [`ArgmaxBenchmark`](@ref).
"""
function ArgmaxBenchmark(; instance_dim::Int=10, nb_features::Int=5, seed=nothing)
    encoder_weights = randn(Utils.make_rng(seed), Float32, 1, nb_features)
    return ArgmaxBenchmark(instance_dim, nb_features, encoder_weights)
end

function Utils.is_minimization_problem(::ArgmaxBenchmark)
    return false
end

"""
$TYPEDSIGNATURES

One-hot encoding of the argmax function.
"""
function one_hot_argmax(z::AbstractVector{R}; kwargs...) where {R<:Real}
    e = zeros(R, length(z))
    e[argmax(z)] = one(R)
    return e
end

"""
$TYPEDSIGNATURES

Return an argmax maximizer.
"""
function Utils.generate_maximizer(::ArgmaxBenchmark)
    return one_hot_argmax
end

"""
$TYPEDSIGNATURES

Generate a data sample for the argmax benchmark.
This function generates a random feature matrix, computes the costs using the encoder,
and adds noise to the costs before computing a target solution.
"""
function Utils.generate_sample(
    bench::ArgmaxBenchmark, rng::AbstractRNG; noise_std::Float32=0.0f0
)
    (; instance_dim, nb_features, encoder_weights) = bench
    features = randn(rng, Float32, nb_features, instance_dim)
    θ_true = vec(encoder_weights * features)
    noisy_y_true = one_hot_argmax(θ_true + noise_std * randn(rng, Float32, instance_dim))
    return DataSample(; x=features, θ=θ_true, y=noisy_y_true)
end

"""
$TYPEDSIGNATURES

Returns a Lux model architecture (single linear layer) for the argmax benchmark.
"""
function Utils.generate_statistical_model(bench::ArgmaxBenchmark)
    (; nb_features) = bench
    return Chain(Dense(nb_features => 1; use_bias=false), WrappedFunction(vec))
end

export ArgmaxBenchmark

end
