module DynamicReplenishment

using ..Utils

using JuMP:
    Model,
    @variable,
    @objective,
    @constraint,
    optimize!,
    value,
    fix,
    primal_status,
    objective_value,
    set_silent,
    MOI,
    AffExpr,
    set_attribute
using Random: Random, AbstractRNG, seed!, randperm, Xoshiro
using Distributions: Poisson, Uniform, Gumbel
using Flux: Chain, Dense, @layer, softplus, relu
using InferOpt: LinearMaximizer
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using LinearAlgebra: dot, I
using Statistics: mean, quantile, std
using StatsBase: ZScoreTransform, fit, transform
"""
$TYPEDEF

Benchmark for a replenishment (retail) problem with production constraints.
Items are chosen according to a a given customer choice model which is endogenous.

# Fields
$TYPEDFIELDS
"""
struct DynamicReplenishmentBenchmark{M} <: AbstractDynamicBenchmark{true}
    "customer choice model (price, features)"
    customer_choice_model::M
    "Poisson arrival rate of customers"
    λ::Float64
    "number of items"
    N::Int
    "dimension of feature vectors (in addition to price: number of objects)"
    d::Int
    "Coupling matrix for quota constraints (nb_constraints x N)"
    constraints_matrix::Matrix{Int}
    "quotas for each constraint at each time step (max_steps x nb_constraints)"
    quotas::Matrix{Int}
    "Lower stock bound"
    stock_inf::Int
    "Upper stock bound"
    stock_sup::Int
    "upper bound of same item in stock"
    ub_same_item::Int
    "delivery delay in days"
    delivery_delay::Int
    "prices of the items (N)"
    prices::Vector{Float64}
    "items' features (d x N)"
    features::Matrix{Float64}
    "cost of virtual stock (N)"
    virtual_stock_cost::Vector{Float64}
    "cost of physical stock (N)"
    physical_stock_cost::Vector{Float64}
    "over stock bound cost"
    over_stock_bound_cost::Float64
    "number of steps per episode"
    max_steps::Int
    "max quota per time step per item (max_steps x N)"
    max_quotas::Matrix{Int}
    "static utilities of the items from the customer choice model (N)"
    static_utilities::Vector{Float64}
end

"""
    DynamicReplenishmentBenchmark(;
        N=10,
        λ=15,
        d=5,
        nb_constraints=2,
        stock_inf=0,
        stock_sup=30,
        ub_same_item=30,
        delivery_delay=3,
        max_steps=10
    )

Constructor for [`DynamicReplenishmentBenchmark`](@ref).
By default, the benchmark has 10 items, feature dimension 5 (+1 for price), 10 steps per
episode, a simple linear customer choice model (all weights are negative), a poisson arrival of 15 customer per time step, and is endogenous. It generates
- random prices uniformly in [1, 10]
- random features uniformly in [-10, 10]
- stock costs are dependant on the price
The user can choose between
- only providing a number of constraints, in which case the constructor generates a random constraints matrix and random quotas
- providing both a constraints matrix and quotas, in which case the constructor uses them as is.
For quotas, the user can choose between fixed quotas (same for all time steps) or random quotas (different for each time step).
"""
function DynamicReplenishmentBenchmark(;
    N::Int=10,
    λ::Int=15,
    d::Int=5,
    nb_constraints::Int=2,
    constraints_matrix=nothing,
    quotas=nothing,
    stock_inf::Int=0,
    stock_sup::Int=30,
    ub_same_item::Int=30,
    delivery_delay::Int=3,
    max_steps::Int=10,
    customer_choice_model=nothing,
    seed=nothing,
    rng=Xoshiro(seed),
)
    if constraints_matrix === nothing || quotas === nothing
        if constraints_matrix !== nothing || quotas !== nothing
            @warn "If either constraints_matrix or quotas is provided, both must be provided. Generating random constraints and quotas."
        end
        constraints_matrix = rand(rng, 0:1, nb_constraints, N)
        quotas = rand(rng, 10:30, max_steps, nb_constraints)
    else
        @assert size(constraints_matrix, 1) == size(quotas, 2) "The number of constraints in the constraints matrix must match the number of columns in the quotas matrix."
        @assert size(constraints_matrix, 2) == N "The number of items ($N) must match the number of columns ($(size(constraints_matrix, 2))) in the constraints matrix."
        @assert size(quotas, 1) == max_steps "The number of steps ($max_steps) must match the number of rows ($(size(quotas, 1))) in the quotas matrix."
        nb_constraints = size(constraints_matrix, 1)
    end

    constraints_matrix = vcat(constraints_matrix, I)
    quotas = hcat([vcat(quotas[t, :], fill(ub_same_item, N)) for t in 1:max_steps]...)'

    prices = rand(rng, Uniform(1.0, 10.0), N)
    features = rand(rng, Uniform(-10.0, 10.0), (d, N))
    if customer_choice_model === nothing
        price_w = rand(rng, Uniform(-1.0, -0.7), 1)
        features_w = rand(rng, Uniform(-0.8, -0.1), d)
        customer_choice_model = Chain(Dense(reshape(vcat(price_w, features_w), 1, :)), vec)
    else
        try
            customer_choice_model(rand(rng, d + 1, N))
        catch e
            throw(
                ArgumentError(
                    "customer_choice_model is incompatible with d=$d (expected input of size (d+1, N)): $e",
                ),
            )
        end
    end
    full_features = vcat(prices', features)   # (d+1, N)
    dt = fit(ZScoreTransform, full_features; dims=2)
    full_features = transform(dt, full_features)
    static_utilities = customer_choice_model(full_features)
    # add no purchase option
    static_utilities = vcat(static_utilities, 0.0)

    virtual_stock_cost = prices ./ (max_steps * 10)
    physical_stock_cost = prices ./ (max_steps * 5)
    over_stock_bound_cost = maximum(prices)
    max_quotas = Matrix{Float64}(undef, max_steps, N)
    for i in 1:N, t in 1:max_steps
        max_quotas[t, i] = minimum(
            quotas[t, c] for
            c in axes(constraints_matrix, 1) if constraints_matrix[c, i] == 1
        )
    end

    return DynamicReplenishmentBenchmark{typeof(customer_choice_model)}(
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
        prices,
        features,
        virtual_stock_cost,
        physical_stock_cost,
        over_stock_bound_cost,
        max_steps,
        max_quotas,
        static_utilities,
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
max_quotas(b::DynamicReplenishmentBenchmark) = b.max_quotas

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

Creates a random environment for the dynamic replenishment benchmark using the provided random number generator.
"""
function Utils.build_environment(
    b::DynamicReplenishmentBenchmark, rng::AbstractRNG; kwargs...
)
    return Environment(b, rng)
end

"""
$TYPEDEF

Callable wrapping [`replenishment_problem`](@ref) with a fixed `model_builder`, so it can
be passed to `LinearMaximizer` without a closure.
"""
struct MaximizerProblem{M}
    model_builder::M
end
function (p::MaximizerProblem)(Θ; kwargs...)
    return replenishment_problem(Θ; kwargs..., model_builder=p.model_builder)
end

function Utils.generate_maximizer(
    ::DynamicReplenishmentBenchmark; model_builder=highs_model
)
    return LinearMaximizer(MaximizerProblem(model_builder); g)
end

"""
$TYPEDEF

Callable wrapping [`anticipative_solver`](@ref) with a fixed `model_builder`, returned by
[`Utils.generate_anticipative_solver`](@ref).
"""
struct AnticipativeSolverCall{M}
    model_builder::M
end
function (s::AnticipativeSolverCall)(
    env::Utils.SeededEnvironment; reset_env=false, kwargs...
)
    _, trajectory = anticipative_solver(
        env.env, env.rng; reset_env, kwargs..., model_builder=s.model_builder
    )
    return trajectory
end

function Utils.generate_anticipative_solver(
    ::DynamicReplenishmentBenchmark; model_builder=highs_model
)
    return AnticipativeSolverCall(model_builder)
end

"""
$TYPEDEF

Callable wrapping [`anticipative_solver`](@ref) (scenario-conditioned) with a fixed
`model_builder`, returned by [`Utils.generate_parametric_anticipative_solver`](@ref).
"""
struct ParametricAnticipativeSolverCall{M}
    model_builder::M
end
function (s::ParametricAnticipativeSolverCall)(
    θ, scenario::Scenario, env::Utils.SeededEnvironment; reset_env=true, kwargs...
)
    # reset_env && Utils.reset_to_initial!(env)
    _, trajectory = anticipative_solver(
        env.env,
        env.rng,
        scenario;
        reset_env=false,
        θ,
        kwargs...,
        model_builder=s.model_builder,
    )
    return trajectory
end

function Utils.generate_parametric_anticipative_solver(
    ::DynamicReplenishmentBenchmark; model_builder=highs_model
)
    return ParametricAnticipativeSolverCall(model_builder)
end

"""
$TYPEDSIGNATURES

Returns two policies for the dynamic replenishment benchmark:
- `Greedy`: "policy that replenishes items in decreasing price order"
- `Random`: "Policy that replenishes items in a random order with random quantities"
"""
function Utils.generate_baseline_policies(
    ::DynamicReplenishmentBenchmark; model_builder=highs_model, kwargs...
)
    greedy = Policy(
        "Greedy", "policy that replenishes items in decreasing price order", greedy_policy
    )
    random = Policy(
        "Random",
        "Policy that replenishes items in a random order with random quantities",
        random_policy,
    )
    lazy = Policy("Lazy", "Policy that replenishes nothing", lazy_policy)
    saa = Policy(
        "SAA",
        "Policy that solves a sample average approximation problem.",
        SAAPolicyCall(model_builder; kwargs...),
    )
    return (; greedy, random, lazy, saa)
end

export DynamicReplenishmentBenchmark

end
