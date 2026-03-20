module ContextualStochasticArgmax

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS
using Flux: Dense
using Random: Random, AbstractRNG, MersenneTwister

function one_hot_argmax(z::AbstractVector{R}; kwargs...) where {R<:Real}
    e = zeros(R, length(z))
    e[argmax(z)] = one(R)
    return e
end

"""
$TYPEDEF

Minimal contextual stochastic argmax benchmark.

Per instance: `c_base ~ U[0,1]^n` (base utility, part of instance kwargs and base features).
Per context draw: `x_raw ~ N(0, I_d)` (observable context). Features: `x = [c_base; x_raw]`.
Per scenario: `ξ = c_base + W * x_raw + noise`, `noise ~ N(0, noise_std² I)`.
The learner sees `x` and must predict `θ̂` so that `argmax(θ̂)` ≈ `argmax(ξ)`.

A linear model `Dense(n+d → n; bias=false)` can exactly recover `[I | W]`.

# Fields
$TYPEDFIELDS
"""
struct ContextualStochasticArgmaxBenchmark{M<:AbstractMatrix} <:
       AbstractStochasticBenchmark{true}
    "number of items (argmax dimension)"
    n::Int
    "number of context features"
    d::Int
    "fixed perturbation matrix W ∈ R^{n×d}, unknown to the learner"
    W::M
    "noise std for scenario draws"
    noise_std::Float32
end

function ContextualStochasticArgmaxBenchmark(;
    n::Int=10, d::Int=5, noise_std::Float32=0.1f0, seed=nothing
)
    rng = MersenneTwister(seed)
    W = randn(rng, Float32, n, d)
    return ContextualStochasticArgmaxBenchmark(n, d, W, noise_std)
end

Utils.is_minimization_problem(::ContextualStochasticArgmaxBenchmark) = false
Utils.generate_maximizer(::ContextualStochasticArgmaxBenchmark) = one_hot_argmax

# c_base: base features (in x) and solver kwarg (in instance_kwargs for generate_scenario)
function Utils.generate_instance(
    bench::ContextualStochasticArgmaxBenchmark, rng::AbstractRNG; kwargs...
)
    c_base = rand(rng, Float32, bench.n)
    return DataSample(; x=c_base, c_base=c_base)
end

# Enriches instance_sample: x = [c_base; x_raw], x_raw in extra for generate_scenario
function Utils.generate_context(
    bench::ContextualStochasticArgmaxBenchmark,
    rng::AbstractRNG,
    instance_sample::DataSample,
)
    x_raw = randn(rng, Float32, bench.d)
    return DataSample(;
        x=vcat(instance_sample.x, x_raw),
        instance_sample.instance_kwargs...,
        extra=(; x_raw),
    )
end

# ξ = c_base + W * x_raw + noise  (c_base from instance_kwargs, x_raw from ctx.extra)
function Utils.generate_scenario(
    bench::ContextualStochasticArgmaxBenchmark,
    rng::AbstractRNG;
    c_base::AbstractVector,
    x_raw::AbstractVector,
    kwargs...,
)
    θ_true = c_base + bench.W * x_raw
    return θ_true + bench.noise_std * randn(rng, Float32, bench.n)
end

function Utils.generate_statistical_model(
    bench::ContextualStochasticArgmaxBenchmark; seed=nothing
)
    Random.seed!(seed)
    return Dense(bench.n + bench.d => bench.n; bias=false)
end

export ContextualStochasticArgmaxBenchmark

end
