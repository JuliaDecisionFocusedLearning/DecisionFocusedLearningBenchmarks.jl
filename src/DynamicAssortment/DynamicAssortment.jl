module DynamicAssortment

using ..Utils

using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES, SIGNATURES
using Distributions: Uniform, Categorical
using Flux: Chain, Dense
using LinearAlgebra: dot
using Random: Random, AbstractRNG, MersenneTwister
using Statistics: mean

using Combinatorics: combinations

"""
$TYPEDEF

Benchmark for the dynamic assortment problem.

# Fields
$TYPEDFIELDS

Reference: <https://arxiv.org/abs/2505.19053>
"""
struct DynamicAssortmentBenchmark{exogenous,M} <: AbstractDynamicBenchmark{exogenous}
    "customer choice model (price, hype, saturation, and features)"
    customer_choice_model::M
    "number of items"
    N::Int
    "dimension of feature vectors (in addition to hype, satisfaction, and price)"
    d::Int
    "assortment size constraint"
    K::Int
    "number of steps per episode"
    max_steps::Int
end

"""
    DynamicAssortmentBenchmark(;
        N=20,
        d=2,
        K=4,
        max_steps=80,
        customer_choice_model=Chain(Dense([-0.8 0.6 -0.4 0.3 0.5]), vec),
        exogenous=false
    )

Constructor for [`DynamicAssortmentBenchmark`](@ref).
By default, the benchmark has 20 items, feature dimension 2, assortment size 4, 80 steps per
episode, a simple linear customer choice model, and is endogenous.
"""

function DynamicAssortmentBenchmark(;
    N=20,
    d=2,
    K=4,
    max_steps=80,
    customer_choice_model=Chain(
        Dense(hcat([-0.8 0.6 -0.4], reshape([0.3 + 0.2 * (i - 1) for i in 1:d], 1, d))), vec
    ),
    exogenous=false,
)
    return DynamicAssortmentBenchmark{exogenous,typeof(customer_choice_model)}(
        customer_choice_model, N, d, K, max_steps
    )
end

# Accessor functions
customer_choice_model(b::DynamicAssortmentBenchmark) = b.customer_choice_model
item_count(b::DynamicAssortmentBenchmark) = b.N
feature_count(b::DynamicAssortmentBenchmark) = b.d
assortment_size(b::DynamicAssortmentBenchmark) = b.K
max_steps(b::DynamicAssortmentBenchmark) = b.max_steps

include("instance.jl")
include("environment.jl")
include("policies.jl")

"""
$TYPEDSIGNATURES

Outputs a data sample containing an [`Instance`](@ref).
"""
function Utils.generate_sample(
    b::DynamicAssortmentBenchmark, rng::AbstractRNG=MersenneTwister(0)
)
    return DataSample(; instance=Instance(b, rng))
end

"""
$TYPEDSIGNATURES

Generates a statistical model for the dynamic assortment benchmark.
The model is a small neural network with one hidden layer of size 5 and no activation function.
"""
function Utils.generate_statistical_model(b::DynamicAssortmentBenchmark; seed=nothing)
    Random.seed!(seed)
    d = feature_count(b)
    return Chain(Dense(d + 8 => 5), Dense(5 => 1), vec)
end

"""
$TYPEDSIGNATURES

Outputs a top k maximizer, with k being the assortment size of the benchmark.
"""
function Utils.generate_maximizer(b::DynamicAssortmentBenchmark)
    return TopKMaximizer(assortment_size(b))
end

"""
$TYPEDSIGNATURES

Creates an [`Environment`](@ref) from an [`Instance`](@ref) of the dynamic assortment benchmark.
The seed of the environment is randomly generated using the provided random number generator.
"""
function Utils.generate_environment(
    ::DynamicAssortmentBenchmark, instance::Instance, rng::AbstractRNG; kwargs...
)
    seed = rand(rng, 1:typemax(Int))
    return Environment(instance; seed)
end

"""
$TYPEDSIGNATURES

Returns two policies for the dynamic assortment benchmark:
- `Greedy`: selects the assortment containing items with the highest prices
- `Expert`: selects the assortment with the highest expected revenue (through brute-force enumeration)
"""
function Utils.generate_policies(::DynamicAssortmentBenchmark)
    greedy = Policy(
        "Greedy",
        "policy that selects the assortment with items with the highest prices",
        greedy_policy,
    )
    expert = Policy(
        "Expert",
        "policy that selects the assortment with the highest expected revenue",
        expert_policy,
    )
    return (expert, greedy)
end

export DynamicAssortmentBenchmark

end
