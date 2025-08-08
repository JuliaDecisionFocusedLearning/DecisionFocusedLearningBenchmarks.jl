module DynamicAssortment

using ..Utils

using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Distributions: Uniform, Categorical
using LinearAlgebra: dot
using Random: Random, AbstractRNG, MersenneTwister
using Statistics: mean

using Flux: Chain, Dense
using Combinatorics: combinations

"""
$TYPEDEF

Benchmark for the dynamic assortment problem.

# Fields
$TYPEDFIELDS
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

function DynamicAssortmentBenchmark(;
    N=20,
    d=2,
    K=4,
    max_steps=80,
    customer_choice_model=Chain(Dense([-0.8 0.6 -0.4 0.3 0.5]), vec),
    exogenous=false,
)
    return DynamicAssortmentBenchmark{exogenous,typeof(customer_choice_model)}(
        customer_choice_model, N, d, K, max_steps
    )
end

include("instance.jl")
include("environment.jl")
include("policies.jl")

customer_choice_model(b::DynamicAssortmentBenchmark) = b.customer_choice_model
item_count(b::DynamicAssortmentBenchmark) = b.N
feature_count(b::DynamicAssortmentBenchmark) = b.d
assortment_size(b::DynamicAssortmentBenchmark) = b.K
max_steps(b::DynamicAssortmentBenchmark) = b.max_steps

function Utils.generate_sample(
    b::DynamicAssortmentBenchmark, rng::AbstractRNG=MersenneTwister(0)
)
    return DataSample(; instance=Instance(b, rng))
end

function Utils.generate_statistical_model(b::DynamicAssortmentBenchmark; seed=nothing)
    Random.seed!(seed)
    d = feature_count(b)
    return Chain(Dense(d + 8 => 5), Dense(5 => 1), vec)
end

function Utils.generate_maximizer(b::DynamicAssortmentBenchmark)
    return TopKMaximizer(assortment_size(b))
end

function Utils.generate_environment(
    ::DynamicAssortmentBenchmark, instance::Instance, rng::AbstractRNG; kwargs...
)
    seed = rand(rng, 1:typemax(Int))
    return Environment(instance; seed)
end

function Utils.generate_policies(b::DynamicAssortmentBenchmark)
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
