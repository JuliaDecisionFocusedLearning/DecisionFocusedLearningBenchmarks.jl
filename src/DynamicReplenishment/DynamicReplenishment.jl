module DynamicReplenishment

# Write your package code here.
using ..Utils

using Combinatorics
# using Gurobi
using IterTools
using JuMP
using Random: Random, AbstractRNG, MersenneTwister, seed!, randperm
using Distributions
using Flux: Chain, Dense, @layer, softplus, relu
using InferOpt: LinearMaximizer
using SCIP
# using HiGHS
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using LinearAlgebra: dot, I

"""
$TYPEDEF

Benchmark for a replenishment problem with production constraints.
Items are chosen according to a agiven customer choice model which is endogenous.

# Fields
$TYPEDFIELDS
"""
struct DynamicReplenishmentBenchmark{exogenous,M} <: AbstractDynamicBenchmark{exogenous}
    "customer choice model (price, mean days one lot features)"
    customer_choice_model::M
    "Poisson arrival rate of customers"
    λ::Float64
    "number of items"
    N::Int
    "dimension of feature vectors (in addition to price: number of objects)"
    d::Int
    "Coupling matrix for quota constraints"
    constraints_matrix::Matrix{Int}
    "quotas for each constraint at each time step"
    quotas::Matrix{Int}
    "Lower stock bound"
    stock_inf::Int
    "Upper stock bound"
    stock_sup::Int
    "upper bound of same archetype in stock"
    ub_same_item::Int
    "delivery delay in days"
    delivery_delay::Int
    "Upper bound for stock of same item for the CO layer"
    UB_item::Int
    "prices of the items"
    prices::Vector{Float64}
    "items' features (d x N matrix)"
    features::Matrix{Float64}
    "cost of virtual stock"
    virtual_stock_cost::Vector{Float64}
    "cost of physical stock"
    physical_stock_cost::Vector{Float64}
    "over stock bound cost"
    over_stock_bound_cost::Float64
    "number of steps per episode"
    max_steps::Int
end

"""
    DynamicReplenishmentBenchmark(;
        N=10,
        λ=15,
        d=5,
        constraints_matrix=[1 1 1 1 1 0 0 0 0 0; 0 0 0 0 0 1 1 1 1 1],
        quotas=[30, 30],
        stock_inf=0,
        stock_sup=50,
        ub_same_item=10,
        delivery_delay=3,
        max_steps=10
    )
end

Constructor for [`DynamicReplenishmentBenchmark`](@ref).
By default, the benchmark has 10 items, feature dimension 5 (+1 for price), 10 steps per
episode, a simple linear customer choice model (all weights are negative), a poisson arrival of 15 customer per time step, and is endogenous.
- random constraint matrix with 2 constraints
- random prices uniformly in [1, 10]
- random features uniformly in [-10, 10]
- stock costs are dependant on the price
"""

function DynamicReplenishmentBenchmark(;
    N=10,
    λ=15,
    d=5,
    constraints_matrix=vcat(
        [i <= N ÷ 2 ? 1 : 0 for _ in 1:1, i in 1:N],
        [i <= N ÷ 2 ? 0 : 1 for _ in 1:1, i in 1:N],
    ),
    quotas=[30, 30],
    stock_inf=0,
    stock_sup=50,
    ub_same_item=10,
    delivery_delay=3,
    max_steps=10,
    UB_item=30,
    customer_choice_model=Chain(Dense([-0.8 -0.4 -0.3 -0.3 -0.3 -0.1]), vec),
    rng=MersenneTwister(0),
)
    UB_item = max(UB_item, ub_same_item, maximum(quotas))
    constraints_matrix = vcat(constraints_matrix, Matrix(1I, N, N))
    quotas = hcat([vcat(quotas, fill(ub_same_item, N)) for _ in 1:max_steps]...)'
    prices = vcat(rand(rng, Uniform(1.0, 10.0), N))
    features = rand(rng, Uniform(-10.0, 10.0), (d, N))
    virtual_stock_cost = prices ./ (max_steps * 10)
    physical_stock_cost = prices ./ (max_steps * 5)
    over_stock_bound_cost = maximum(prices) * 10
    return DynamicReplenishmentBenchmark{false,typeof(customer_choice_model)}(
        customer_choice_model,
        λ,
        N,
        d,
        constraints_matrix,
        quotas,
        stock_inf,
        stock_sup,
        ub_same_item,
        delivery_delay,
        UB_item,
        prices,
        features,
        virtual_stock_cost,
        physical_stock_cost,
        over_stock_bound_cost,
        max_steps,
    )
end

# Accessor functions
customer_choice_model(b::DynamicReplenishmentBenchmark) = b.customer_choice_model
poisson_arrival_rate(b::DynamicReplenishmentBenchmark) = b.λ
item_count(b::DynamicReplenishmentBenchmark) = b.N
feature_count(b::DynamicReplenishmentBenchmark) = b.d
max_steps(b::DynamicReplenishmentBenchmark) = b.max_steps
constraints_matrix(b::DynamicReplenishmentBenchmark) = b.constraints_matrix
quotas(b::DynamicReplenishmentBenchmark) = b.quotas
stock_inf(b::DynamicReplenishmentBenchmark) = b.stock_inf
stock_sup(b::DynamicReplenishmentBenchmark) = b.stock_sup
ub_same_item(b::DynamicReplenishmentBenchmark) = b.ub_same_item
delivery_delay(b::DynamicReplenishmentBenchmark) = b.delivery_delay
prices(b::DynamicReplenishmentBenchmark) = b.prices
features(b::DynamicReplenishmentBenchmark) = b.features
virtual_stock_cost(b::DynamicReplenishmentBenchmark) = b.virtual_stock_cost
physical_stock_cost(b::DynamicReplenishmentBenchmark) = b.physical_stock_cost
over_stock_bound_cost(b::DynamicReplenishmentBenchmark) = b.over_stock_bound_cost
nb_constraints(b::DynamicReplenishmentBenchmark) = size(b.constraints_matrix, 1)
UB_item(b::DynamicReplenishmentBenchmark) = b.UB_item

function max_quota_per_step_per_item(b::DynamicReplenishmentBenchmark)
    T = max_steps(b)
    N = item_count(b)
    cons_mat = constraints_matrix(b)
    q = quotas(b)
    max_quotas = Matrix{Float64}(undef, T, N)
    for i in 1:N
        if sum(cons_mat[:, i]) == 0
            max_quotas[:, i] .= ub_same_item(b)
        else
            for t in 1:T
                max_quotas[t, i] = min(
                    minimum([q[t, c] for c in 1:nb_constraints(b) if cons_mat[c, i] == 1]),
                    ub_same_item(b),
                )
            end
        end
    end
    return max_quotas
end

include("utils.jl")

include("state.jl")
include("scenario.jl")
include("environment.jl")
include("statistical_model.jl")
include("policies.jl")
include("maximizer.jl")
include("anticipative_solver.jl")
include("features.jl")

"""
$TYPEDSIGNATURES

Creates an environment for the dynamic replenishment benchmark.
The seed of the environment is randomly generated using the provided random number generator.
"""
function Utils.generate_environment(
    b::DynamicReplenishmentBenchmark, rng::AbstractRNG; kwargs...
)
    seed = rand(rng, 1:typemax(Int))
    return Environment(b; seed=seed, rng=rng)
end

"""
$TYPEDSIGNATURES

"""
function Utils.generate_maximizer(::DynamicReplenishmentBenchmark)
    return LinearMaximizer(replenishment_problem; g)
end

function Utils.generate_anticipative_solver(::DynamicReplenishmentBenchmark)
    return (env; reset_env=true, kwargs...) -> begin
        _, trajectory = anticipative_solver(env; reset_env, kwargs...)
        return trajectory
    end
end

"""
$TYPEDSIGNATURES

Returns two policies for the dynamic replenishment benchmark:
- `Greedy`: "policy that replenishes items in decreasing price order"
- `Random`: "Policy that replenishes items in a random order with random quantities"
"""
function Utils.generate_baseline_policies(::DynamicReplenishmentBenchmark)
    greedy = Policy(
        "Greedy", "policy that replenishes items in decreasing price order", greedy_policy
    )
    random = Policy(
        "Random",
        "Policy that replenishes items in a random order with random quantities",
        random_policy,
    )
    return (; greedy, random)
end

export DynamicReplenishmentBenchmark

end
