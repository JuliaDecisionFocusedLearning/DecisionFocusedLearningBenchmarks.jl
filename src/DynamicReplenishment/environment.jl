"""
$TYPEDEF

Environment for the Dynamic Replenishment problem.

# Fields
$TYPEDFIELDS
"""
@kwdef mutable struct Environment{
    B<:DynamicReplenishmentBenchmark,S<:DRPState,R<:AbstractRNG,SS
} <: Utils.AbstractEnvironment
    "associated benchmark"
    config::B
    "current state"
    state::S
    "scenario the environment will use when not given a specific one"
    scenario::Scenario
    "initial stock"
    stock_ini::Vector{Int}
    "random number generator"
    rng::R
    "seed for the environment"
    seed::SS
end

# Accessor functions
customer_choice_model(env::Environment) = customer_choice_model(env.config)
poisson_arrival_rate(env::Environment) = poisson_arrival_rate(env.config)
item_count(env::Environment) = item_count(env.config)
feature_count(env::Environment) = feature_count(env.config)
max_steps(env::Environment) = max_steps(env.config)
constraints_matrix(env::Environment) = constraints_matrix(env.config)
quotas(env::Environment) = quotas(env.config)
stock_inf(env::Environment) = stock_inf(env.config)
stock_sup(env::Environment) = stock_sup(env.config)
ub_same_item(env::Environment) = ub_same_item(env.config)
delivery_delay(env::Environment) = delivery_delay(env.config)
prices(env::Environment) = prices(env.config)
features(env::Environment) = features(env.config)
virtual_stock_cost(env::Environment) = virtual_stock_cost(env.config)
physical_stock_cost(env::Environment) = physical_stock_cost(env.config)
over_stock_bound_cost(env::Environment) = over_stock_bound_cost(env.config)
max_quotas(env::Environment) = max_quotas(env.config)

current_epoch(env::Environment) = current_epoch(env.state)
stock_ini(env::Environment) = env.stock_ini
stock(env::Environment) = stock(env.state)
ub_per_item(env::Environment) = ub_per_item(env.state)

"""
$TYPEDSIGNATURES

Creates an [`Environment`](@ref) from an instance of the dynamic replenishment benchmark.
Initialize the initial stock to Uniform(0, 10).
"""
function Environment(
    config::DynamicReplenishmentBenchmark;
    seed=0,
    rng::AbstractRNG=MersenneTwister(seed),
    stock_ini=rand(rng, 0:10, item_count(config)),
)
    N = item_count(config)
    scenario = Utils.generate_scenario(config; seed=seed, rng=rng)
    initial_state = DRPState(config, stock_ini)
    return Environment(;
        config, state=initial_state, scenario, stock_ini, rng=rng, seed=seed
    )
end

function Environment(
    config::DynamicReplenishmentBenchmark,
    scenario::Scenario;
    stock_ini=rand(rng, 0:10, item_count(config)),
    seed=0,
    rng::AbstractRNG=MersenneTwister(seed),
)
    initial_state = DRPState(config, stock_ini)
    return Environment(;
        config, state=initial_state, scenario, stock_ini, rng=rng, seed=seed
    )
end

Utils.get_seed(env::Environment) = env.seed

"""
$TYPEDSIGNATURES

Get the current state of the environment.
"""
function Utils.observe(env::Environment)
    return compute_features(env.state), env.state
end

"""
$TYPEDSIGNATURES

Check if the episode is terminated, i.e. if the current epoch is the last one.
The +1 comes from the initial state which is considered as epoch 0 (but labeled 1).
"""
Utils.is_terminated(env::Environment) = current_epoch(env) > max_steps(env)

"""
$TYPEDSIGNATURES

Reset the environment to its initial state.
Also reset the rng to `seed` if `reset_rng` is set to true.
"""
function Utils.reset!(env::Environment; seed=get_seed(env), reset_rng=false)
    if reset_rng
        Random.seed!(env.rng, seed)
    end
    env.scenario = Utils.generate_scenario(env.config; seed, rng=env.rng)
    reset_state!(env.state)
    return nothing
end

"""
$TYPEDSIGNATURES

Apply the replenishment to the stock, apply the sales and increase time.
"""
function Utils.step!(env::Environment, replenishment)
    replenishment = replenishment
    @assert !Utils.is_terminated(env) "Environment is terminated, cannot act!"
    apply_replenishment!(env.state, replenishment)
    delta_cost = apply_sales!(env.state; env.scenario[current_epoch(env)]...)
    add_customers!(env.state; env.scenario[current_epoch(env)]...)
    if current_epoch(env) < max_steps(env)
        env.state.ub_per_item =
            env.state.stock .+ max_quotas(env.config)[current_epoch(env) + 1]
    end
    env.state.current_epoch += 1
    return delta_cost
end