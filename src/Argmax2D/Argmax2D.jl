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

Custom constructor for [`ArgmaxBenchmark`](@ref).
"""
function Argmax2DBenchmark(; nb_features::Int=5, seed=nothing, polytope_vertex_range=[6])
    Random.seed!(seed)
    model = Chain(Dense(nb_features => 2; bias=false), vec)
    return Argmax2DBenchmark(nb_features, model, polytope_vertex_range)
end

maximizer(θ; instance) = instance[argmax(dot(θ, v) for v in instance)]

function Utils.generate_dataset(
    bench::Argmax2DBenchmark, dataset_size=10; seed=nothing, rng=MersenneTwister(seed)
)
    (; nb_features, encoder, polytope_vertex_range) = bench
    return map(1:dataset_size) do _
        x = randn(rng, nb_features)
        θ_true = encoder(x)
        θ_true ./= 2 * norm(θ_true)
        instance = build_polytope(rand(rng, polytope_vertex_range); shift=rand(rng))
        y_true = maximizer(θ_true; instance)
        return DataSample(; x=x, θ_true=θ_true, y_true=y_true, instance=instance)
    end
end

Utils.generate_maximizer(::Argmax2DBenchmark) = maximizer

function Utils.generate_statistical_model(
    bench::Argmax2DBenchmark; seed=nothing, rng=MersenneTwister(seed)
)
    Random.seed!(rng, seed)
    (; nb_features) = bench
    model = Chain(Dense(nb_features => 2; bias=false), vec)
    return model
end

function Utils.plot_data(
    ::Argmax2DBenchmark, sample::DataSample; θ_true=sample.θ_true, kwargs...
)
    (; instance) = sample
    pl = init_plot()
    plot_polytope!(pl, instance)
    plot_objective!(pl, θ_true)
    return plot_maximizer!(pl, θ_true, instance, maximizer)
end

export Argmax2DBenchmark

end
