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
    termination_status,
    objective_value,
    set_silent,
    MOI,
    AffExpr,
    set_attribute,
    set_start_value
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
        constraints_matrix=nothing,
        quotas=nothing,
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
    if isnothing(constraints_matrix) || isnothing(quotas)
        if !isnothing(constraints_matrix) || !isnothing(quotas)
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
    quotas = hcat(quotas, fill(ub_same_item, max_steps, N))

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
    max_quotas = Matrix{Int}(undef, max_steps, N)
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

# The objective is a margin net of stock costs, maximized.
Utils.is_minimization_problem(::DynamicReplenishmentBenchmark) = false

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
$TYPEDSIGNATURES

Rebuild an environment sitting at `sample.state` and running on `scenario`, so that the
callable returned by [`Utils.generate_parametric_anticipative_solver`](@ref) can be applied
at any epoch of a stored trajectory.

The state is shared with `sample`, not copied: the solvers reading it must not mutate it.
"""
function Utils.build_environment(
    ::DynamicReplenishmentBenchmark, sample::DataSample, scenario::Scenario
)
    hasproperty(sample, :state) || error(
        "`build_environment` needs the epoch state in `sample.state`, got a DataSample " *
        "with $(propertynames(sample)).",
    )
    state = sample.state
    state isa DRPState ||
        error("`sample.state` should be a `DRPState`, got a $(typeof(state)) instead.")
    return Environment(;
        config=state.config, state=state, scenario=scenario, stock_ini=stock_ini(state)
    )
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
    "relative MIP gap tolerance"
    mip_gap::Float64
    "solver time limit in seconds, `nothing` to solve to optimality"
    time_limit::Union{Float64,Nothing}
end

function AnticipativeSolverCall(
    model_builder::M; mip_gap::Real=0.0, time_limit::Union{Real,Nothing}=nothing
) where {M}
    return AnticipativeSolverCall{M}(
        model_builder,
        Float64(mip_gap),
        isnothing(time_limit) ? nothing : Float64(time_limit),
    )
end

function (s::AnticipativeSolverCall)(
    env::Utils.SeededEnvironment; reset_env=false, kwargs...
)
    _, trajectory = anticipative_solver(
        env.env,
        env.rng;
        reset_env,
        mip_gap=s.mip_gap,
        time_limit=s.time_limit,
        kwargs...,
        model_builder=s.model_builder,
    )
    return trajectory
end

"""
$TYPEDSIGNATURES

Return the anticipative solver for the dynamic replenishment benchmark, as a callable
taking a [`SeededEnvironment`](@ref) and returning the anticipative trajectory.

`mip_gap` and `time_limit` are baked into the returned callable, so that callers that only
hand it an environment (e.g. an imitation-learning expert loop) can still bound how long
each expert call is allowed to run. Both stay overridable per call.

This callable drops the MILP objective value. When that bound is what you need — it is the
reference every optimality gap is measured against — use [`AnticipativePolicy`](@ref)
instead: as an [`AbstractTrajectoryPolicy`](@ref), `rollout!` returns it alongside the
trajectory.
"""
function Utils.generate_anticipative_solver(
    ::DynamicReplenishmentBenchmark;
    model_builder=highs_model,
    mip_gap::Real=0.0,
    time_limit::Union{Real,Nothing}=nothing,
)
    return AnticipativeSolverCall(model_builder; mip_gap, time_limit)
end

"""
$TYPEDEF

Callable wrapping [`anticipative_solver`](@ref) (scenario-conditioned) with a fixed
`model_builder`, returned by [`Utils.generate_parametric_anticipative_solver`](@ref).
"""
struct ParametricAnticipativeSolverCall{M}
    model_builder::M
    "relative MIP gap tolerance"
    mip_gap::Float64
    "solver time limit in seconds, `nothing` to solve to optimality"
    time_limit::Union{Float64,Nothing}
end

function ParametricAnticipativeSolverCall(
    model_builder::M; mip_gap::Real=0.0, time_limit::Union{Real,Nothing}=nothing
) where {M}
    return ParametricAnticipativeSolverCall{M}(
        model_builder,
        Float64(mip_gap),
        isnothing(time_limit) ? nothing : Float64(time_limit),
    )
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
        mip_gap=s.mip_gap,
        time_limit=s.time_limit,
        kwargs...,
        model_builder=s.model_builder,
    )
    return trajectory
end

function Utils.generate_parametric_anticipative_solver(
    ::DynamicReplenishmentBenchmark;
    model_builder=highs_model,
    mip_gap::Real=0.0,
    time_limit::Union{Real,Nothing}=nothing,
)
    return ParametricAnticipativeSolverCall(model_builder; mip_gap, time_limit)
end

"""
$TYPEDSIGNATURES

Returns baseline policies for the dynamic replenishment benchmark: `Greedy`, `Random`,
`Lazy`, `MeanAnticipative`, `MeanAnticipativePerEpoch` and `SAA`.

Both mean policies need `anticipative_results` (anticipative decisions), if empty, they fall back to `Lazy`. 
Remaining keyword arguments go to the SAA
policy.
"""
function Utils.generate_baseline_policies(
    ::DynamicReplenishmentBenchmark;
    model_builder=highs_model,
    anticipative_results::AbstractVector{<:DataSample}=DataSample[],
    order_item::Function=mean_feature_order,
    kwargs...,
)
    greedy = Policy{DynamicReplenishmentBenchmark}(
        "Greedy", "policy that replenishes items in decreasing price order", greedy_policy
    )
    random = Policy{DynamicReplenishmentBenchmark}(
        "Random",
        "Policy that replenishes items in a random order with random quantities",
        random_policy,
    )
    lazy = Policy{DynamicReplenishmentBenchmark}(
        "Lazy", "Policy that replenishes nothing", lazy_policy
    )
    mean_anticipative = Policy{DynamicReplenishmentBenchmark}(
        "MeanAnticipative",
        "Policy that replenishes items in increasing mean feature order, with quantities equal to the mean of the anticipative results over the whole horizon",
        MeanAnticipativePolicyCall(anticipative_results, order_item),
    )
    mean_anticipative_per_epoch = Policy{DynamicReplenishmentBenchmark}(
        "MeanAnticipativePerEpoch",
        "Policy that replenishes items in increasing mean feature order, with quantities equal to the mean of the anticipative results at the current epoch",
        MeanAnticipativePolicyCall(anticipative_results, order_item; per_epoch=true),
    )
    saa = Policy{DynamicReplenishmentBenchmark}(
        "SAA",
        "Policy that solves a sample average approximation problem.",
        SAAPolicyCall(model_builder; kwargs...),
    )
    return (; greedy, random, lazy, mean_anticipative, mean_anticipative_per_epoch, saa)
end

export DynamicReplenishmentBenchmark

end
