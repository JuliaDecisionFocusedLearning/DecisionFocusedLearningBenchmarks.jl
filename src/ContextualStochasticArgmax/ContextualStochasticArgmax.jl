module ContextualStochasticArgmax

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS
using Flux: Dense
using Random: Random, AbstractRNG, MersenneTwister

"""
$TYPEDEF

Minimal contextual stochastic argmax benchmark.

Per instance: `c_base ~ U[0,1]^n` (base utility, stored in `extra` of the instance sample).
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

"""
    generate_instance(::ContextualStochasticArgmaxBenchmark, rng)

Draw `c_base ~ U[0,1]^n` and store it in `extra`. No solver kwargs are needed
(the maximizer is `one_hot_argmax`, which takes no kwargs).
"""
function Utils.generate_instance(
    bench::ContextualStochasticArgmaxBenchmark, rng::AbstractRNG; kwargs...
)
    c_base = rand(rng, Float32, bench.n)
    return DataSample(; extra=(; c_base))
end

"""
    generate_context(::ContextualStochasticArgmaxBenchmark, rng, instance_sample)

Draw `x_raw ~ N(0, I_d)` and return a context sample with:
- `x = [c_base; x_raw]`: full feature vector seen by the ML model.
- `extra = (; c_base, x_raw)`: latents spread into [`generate_scenario`](@ref).
"""
function Utils.generate_context(
    bench::ContextualStochasticArgmaxBenchmark,
    rng::AbstractRNG,
    instance_sample::DataSample,
)
    c_base = instance_sample.c_base
    x_raw = randn(rng, Float32, bench.d)
    return DataSample(; x=vcat(c_base, x_raw), extra=(; x_raw, c_base))
end

"""
    generate_scenario(::ContextualStochasticArgmaxBenchmark, rng; c_base, x_raw, kwargs...)

Draw `ξ = c_base + W * x_raw + noise`, `noise ~ N(0, noise_std² I)`.
`c_base` and `x_raw` are spread from `ctx.extra` by the framework.
"""
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
