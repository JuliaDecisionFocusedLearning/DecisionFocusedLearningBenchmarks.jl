module Maintenance

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

Benchmark for a standard maintenance problem with resource constraints.
Components are identical and degrade independently over time.
A high cost is incurred for each component that reaches the final degradation level. 
A cost is also incurred for maintaining a component. 
The number of simultaneous maintenance operations is limited by a maintenance capacity constraint.

# Fields
$TYPEDFIELDS

"""
struct MaintenanceBenchmark <: AbstractDynamicBenchmark{true}
    "number of components"
    N::Int
    "maximum number of components that can be maintained simultaneously"
    K::Int
    "number of degradation states per component"
    n::Int
    "degradation probability"
    p::Float64
    "failure cost"
    c_f::Float64
    "maintenance cost"
    c_m::Float64
    "number of steps per episode"
    max_steps::Int

    function MaintenanceBenchmark(N, K, n, p, c_f, c_m, max_steps)
        @assert K <= N "number of maintained components $K > number of components $N"
        @assert K >= 0 && N >= 0 "number of components should be positive"
        @assert 0 <= p <= 1 "degradation probability $p is not in [0, 1]"
        # ...
        return new(N, K, n, p, c_f, c_m, max_steps)
    end
end

"""
    MaintenanceBenchmark(;
        N=2,
        K=1,
        n=3,
        p=0.2
        c_f=10.0,
        c_m=3.0,
        max_steps=80,
    )

Constructor for [`MaintenanceBenchmark`](@ref).
By default, the benchmark has 2 components, maintenance capacity 1, number of degradation levels 3, 
degradation probability 0.2, failure cost 10.0, maintenance cost 3.0, 80 steps per episode, and is exogenous.
"""
function MaintenanceBenchmark(; N=2, K=1, n=3, p=0.2, c_f=10.0, c_m=3.0, max_steps=80)
    return MaintenanceBenchmark(N, K, n, p, c_f, c_m, max_steps)
end

# Accessor functions
component_count(b::MaintenanceBenchmark) = b.N
maintenance_capacity(b::MaintenanceBenchmark) = b.K
degradation_levels(b::MaintenanceBenchmark) = b.n
degradation_probability(b::MaintenanceBenchmark) = b.p
failure_cost(b::MaintenanceBenchmark) = b.c_f
maintenance_cost(b::MaintenanceBenchmark) = b.c_m
max_steps(b::MaintenanceBenchmark) = b.max_steps

include("instance.jl")
include("environment.jl")
include("policies.jl")
include("maximizer.jl")

"""
$TYPEDSIGNATURES

Outputs a data sample containing an [`Instance`](@ref).
"""
function Utils.generate_sample(b::MaintenanceBenchmark, rng::AbstractRNG)
    return DataSample(; instance=Instance(b, rng))
end

"""
$TYPEDSIGNATURES

Generates a statistical model for the maintenance benchmark.
The model is a small neural network with one hidden layer no activation function.
"""
function Utils.generate_statistical_model(b::MaintenanceBenchmark; seed=nothing)
    Random.seed!(seed)
    N = component_count(b)
    return Chain(Dense(N => N), Dense(N => N), vec)
end

"""
$TYPEDSIGNATURES

Outputs a top k maximizer, with k being the maintenance capacity of the benchmark.
"""
function Utils.generate_maximizer(b::MaintenanceBenchmark)
    return TopKPositiveMaximizer(maintenance_capacity(b))
end

"""
$TYPEDSIGNATURES

Creates an [`Environment`](@ref) from an [`Instance`](@ref) of the maintenance benchmark.
The seed of the environment is randomly generated using the provided random number generator.
"""
function Utils.generate_environment(
    ::MaintenanceBenchmark, instance::Instance, rng::AbstractRNG; kwargs...
)
    seed = rand(rng, 1:typemax(Int))
    return Environment(instance; seed)
end

"""
$TYPEDSIGNATURES

Returns two policies for the dynamic assortment benchmark:
- `Greedy`: maintains components when they are in the last state before failure, up to the maintenance capacity
"""
function Utils.generate_policies(::MaintenanceBenchmark)
    greedy = Policy(
        "Greedy",
        "policy that maintains components when they are in the last state before failure, up to the maintenance capacity",
        greedy_policy,
    )
    return (greedy,)
end

export MaintenanceBenchmark

end
