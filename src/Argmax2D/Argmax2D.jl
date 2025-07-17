module Argmax2D

using ..Utils
using Colors: Colors
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Flux: Chain, Dense
using LaTeXStrings: @L_str
using LinearAlgebra: dot, norm
using Plots: Plots
using Random: Random, MersenneTwister

include("polytope.jl")

"""
$TYPEDEF

Argmax becnhmark on a 2d polytope.

# Fields
$TYPEDFIELDS
"""
struct Argmax2DBenchmark{E,R} <: AbstractBenchmark
    "number of features"
    nb_features::Int
    "true mapping between features and costs"
    encoder::E
    ""
    polytope_vertex_range::R
end

function Base.show(io::IO, bench::Argmax2DBenchmark)
    (; nb_features) = bench
    return print(io, "Argmax2DBenchmark(nb_features=$nb_features)")
end

"""
$TYPEDSIGNATURES

Custom constructor for [`Argmax2DBenchmark`](@ref).
"""
function Argmax2DBenchmark(; nb_features::Int=5, seed=nothing, polytope_vertex_range=[6])
    Random.seed!(seed)
    model = Dense(nb_features => 2; bias=false)
    return Argmax2DBenchmark(nb_features, model, polytope_vertex_range)
end

Utils.is_minimization_problem(::Argmax2DBenchmark) = false

maximizer(θ; instance, kwargs...) = instance[argmax(dot(θ, v) for v in instance)]

"""
$TYPEDSIGNATURES

Generate a dataset for the [`Argmax2DBenchmark`](@ref).
"""
function Utils.generate_dataset(
    bench::Argmax2DBenchmark, dataset_size=10; seed=nothing, rng=MersenneTwister(seed)
)
    (; nb_features, encoder, polytope_vertex_range) = bench
    return map(1:dataset_size) do _
        x = randn(rng, Float32, nb_features)
        θ_true = encoder(x)
        θ_true ./= 2 * norm(θ_true)
        instance = build_polytope(rand(rng, polytope_vertex_range); shift=rand(rng))
        y_true = maximizer(θ_true; instance)
        return DataSample(; x=x, θ_true=θ_true, y_true=y_true, instance=instance)
    end
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

Generate a statistical model for the [`Argmax2DBenchmark`](@ref).
"""
function Utils.generate_statistical_model(
    bench::Argmax2DBenchmark; seed=nothing, rng=MersenneTwister(seed)
)
    Random.seed!(rng, seed)
    (; nb_features) = bench
    model = Dense(nb_features => 2; bias=false)
    return model
end

function Utils.plot_data(::Argmax2DBenchmark; instance, θ, kwargs...)
    pl = init_plot()
    plot_polytope!(pl, instance)
    plot_objective!(pl, θ)
    return plot_maximizer!(pl, θ, instance, maximizer)
end

"""
$TYPEDSIGNATURES

Plot the data sample for the [`Argmax2DBenchmark`](@ref).
"""
function Utils.plot_data(
    bench::Argmax2DBenchmark,
    sample::DataSample;
    instance=sample.instance,
    θ=sample.θ_true,
    kwargs...,
)
    return Utils.plot_data(bench; instance, θ, kwargs...)
end

export Argmax2DBenchmark

end
