module Argmax2D

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Lux: Dense
using LinearAlgebra: dot, norm
using Random: Random, AbstractRNG

include("polytope.jl")

"""
$TYPEDEF

Argmax becnhmark on a 2d polytope.

# Fields
$TYPEDFIELDS
"""
struct Argmax2DBenchmark{W<:AbstractMatrix,R} <: AbstractStaticBenchmark
    "number of features"
    nb_features::Int
    "true weight matrix mapping features to costs"
    encoder_weights::W
    ""
    polytope_vertex_range::R
end

function Base.show(io::IO, bench::Argmax2DBenchmark)
    (; nb_features) = bench
    return print(io, "Argmax2DBenchmark(nb_features=$nb_features)")
end

Utils.objective_value(::Argmax2DBenchmark, sample::DataSample, y) = dot(sample.θ, y)

"""
$TYPEDSIGNATURES

Custom constructor for [`Argmax2DBenchmark`](@ref).
"""
function Argmax2DBenchmark(; nb_features::Int=5, seed=nothing, polytope_vertex_range=[6])
    encoder_weights = randn(Utils.make_rng(seed), Float32, 2, nb_features)
    return Argmax2DBenchmark(nb_features, encoder_weights, polytope_vertex_range)
end

function Utils.is_minimization_problem(::Argmax2DBenchmark)
    return false
end

maximizer(θ; instance, kwargs...) = instance[argmax(dot(θ, v) for v in instance)]

"""
$TYPEDSIGNATURES

Generate a sample for the [`Argmax2DBenchmark`](@ref).
"""
function Utils.generate_sample(bench::Argmax2DBenchmark, rng::AbstractRNG)
    (; nb_features, encoder_weights, polytope_vertex_range) = bench
    x = randn(rng, Float32, nb_features)
    θ_true = encoder_weights * x
    θ_true ./= 2 * norm(θ_true)
    instance = build_polytope(rand(rng, polytope_vertex_range); shift=rand(rng))
    y_true = maximizer(θ_true; instance)
    return DataSample(; x=x, θ=θ_true, y=y_true, instance=instance)
end

"""
$TYPEDSIGNATURES

Maximizer for the [`Argmax2DBenchmark`](@ref).
"""
function Utils.generate_maximizer(::Argmax2DBenchmark)
    return maximizer
end

"""
$TYPEDSIGNATURES

Returns a Lux model architecture for the Argmax2D benchmark.
"""
function Utils.generate_statistical_model(bench::Argmax2DBenchmark)
    (; nb_features) = bench
    return Dense(nb_features => 2; use_bias=false)
end

export Argmax2DBenchmark

end
