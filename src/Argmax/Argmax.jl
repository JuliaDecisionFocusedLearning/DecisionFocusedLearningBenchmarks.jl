module Argmax

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Flux: Chain, Dense
using Random

"""
$TYPEDEF

Basic benchmark problem with an argmax as the CO algorithm.

# Fields
$TYPEDFIELDS
"""
struct ArgmaxBenchmark{E} <: AbstractBenchmark
    "instances dimension, total number of classes"
    instance_dim::Int
    "number of features"
    nb_features::Int
    "true mapping between features and costs"
    encoder::E
end

function Base.show(io::IO, bench::ArgmaxBenchmark)
    (; instance_dim, nb_features) = bench
    return print(
        io, "ArgmaxBenchmark(instance_dim=$instance_dim, nb_features=$nb_features)"
    )
end

"""
$TYPEDSIGNATURES

Custom constructor for [`ArgmaxBenchmark`](@ref).
"""
function ArgmaxBenchmark(; instance_dim::Int=10, nb_features::Int=5, seed=nothing)
    Random.seed!(seed)
    model = Chain(Dense(nb_features => 1; bias=false), vec)
    return ArgmaxBenchmark(instance_dim, nb_features, model)
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
function Utils.generate_maximizer(bench::ArgmaxBenchmark)
    return one_hot_argmax
end

"""
$TYPEDSIGNATURES

Generate a dataset of labeled instances for the argmax problem.
"""
function Utils.generate_dataset(
    bench::ArgmaxBenchmark, dataset_size::Int=10; seed::Int=0, noise_std=0.0
)
    (; instance_dim, nb_features, encoder) = bench
    rng = MersenneTwister(seed)
    features = [randn(rng, Float32, nb_features, instance_dim) for _ in 1:dataset_size]
    costs = encoder.(features)
    noisy_solutions = [
        one_hot_argmax(θ + noise_std * randn(rng, Float32, instance_dim)) for θ in costs
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
function Utils.generate_statistical_model(bench::ArgmaxBenchmark; seed=nothing)
    (; nb_features) = bench
    Random.seed!(seed)
    return Chain(Dense(nb_features => 1; bias=false), vec)
end

export ArgmaxBenchmark

end
