"""
$TYPEDEF

Environment for the maintenance problem.

# Fields
$TYPEDFIELDS
"""
@kwdef mutable struct Environment{I<:Instance,R<:AbstractRNG,S<:Union{Nothing,Int}} <:
                      Utils.AbstractEnvironment
    "associated instance"
    instance::I
    "current step"
    step::Int
    "degradation state"
    degradation_state::Vector{Int}
    "rng"
    rng::R
    "seed for RNG"
    seed::S
end

"""
$TYPEDSIGNATURES

Creates an [`Environment`](@ref) from an [`Instance`](@ref) of the maintenance benchmark.
"""
function Environment(instance::Instance; seed=0, rng::AbstractRNG=MersenneTwister(seed))
    degradation_state = starting_state(instance)
    env = Environment(;
        instance,
        step=1,
        degradation_state,
        rng=rng,
        seed=seed,
    )
    Utils.reset!(env; reset_rng=true)
    return env
end

component_count(env::Environment) = component_count(env.instance) 
maintenance_capacity(env::Environment) = maintenance_capacity(env.instance)
degradation_levels(env::Environment) = degradation_levels(env.instance)
degradation_probability(env::Environment) = degradation_probability(env.instance)
failure_cost(env::Environment) = failure_cost(env.instance)
maintenance_cost(env::Environment) = maintenance_cost(env.instance)
max_steps(env::Environment) = max_steps(env.instance) 
starting_state(env::Environment) = starting_state(env.instance)


"""
$TYPEDSIGNATURES
Draw random degradations for all components.
"""

function degrad!(env::Environment)
    N = component_count(env) 
    n = degradation_levels(env)
    p = degradation_probability(env)

    for i in 1:N
        if env.degradation_state[i] < n && rand() < p
            env.degradation_state[i] += 1
        end
    end

    return env.degradation_state
end

"""
$TYPEDSIGNATURES
Maintain components.
"""

function maintain!(env::Environment, maintenance::BitVector)
    N = component_count(env) 

    for i in 1:N
        if maintenance[i]
            env.degradation_state[i] = 1
        end
    end

    return env.degradation_state
end

"""
$TYPEDSIGNATURES

Compute maintenance cost.
"""
function maintenance_cost(env::Environment, maintenance::BitVector)
    return maintenance_cost(env) * sum(maintenance)
end

"""
$TYPEDSIGNATURES

Compute degradation cost.
"""
function degradation_cost(env::Environment)
    N = component_count(env) 
    n = degradation_levels(env)
    return failure_cost(env) * count(==(n), env.degradation_state)
end


"""
$TYPEDSIGNATURES

Outputs the seed of the environment.
"""
Utils.get_seed(env::Environment) = env.seed

"""
$TYPEDSIGNATURES

Resets the environment to the initial state:
- reset the rng if `reset_rng` is true
- reset the step to 1
- reset the degradation state to the starting state
"""
function Utils.reset!(env::Environment; reset_rng=false, seed=env.seed)
    reset_rng && Random.seed!(env.rng, seed)
    env.step = 1
    env.degradation_state .= starting_state(env)
    return nothing
end

"""
$TYPEDSIGNATURES

Checks if the environment has reached the maximum number of steps.
"""
function Utils.is_terminated(env::Environment)
    return env.step > max_steps(env)
end

"""
$TYPEDSIGNATURES

Returns features, state tuple.
The features observed by the agent at current step are the degradation states of all components.
It is also the internal state, so we return the same thing twice.

"""
function Utils.observe(env::Environment)
    state = env.degradation_state
    return state, state
end

"""
$TYPEDSIGNATURES

Performs one step in the environment given a maintenance.
Draw random degradations for components that are not maintained.
"""
function Utils.step!(env::Environment, maintenance::BitVector)
    @assert !Utils.is_terminated(env) "Environment is terminated, cannot act!"
    cost = maintenance_cost(env, maintenance) + degradation_cost(env)
    degrad!(env)
    maintain!(env, maintenance)
    env.step += 1
    return cost
end


