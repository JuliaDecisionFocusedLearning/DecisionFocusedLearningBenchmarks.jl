"""
$TYPEDEF

Environment for the maintenance problem.

# Fields
$TYPEDFIELDS
"""
@kwdef mutable struct Environment <: AbstractEnvironment
    "associated instance"
    instance::Instance
    "current step"
    step::Int
    "degradation state"
    degradation_state::Vector{Int}
end

"""
$TYPEDSIGNATURES

Creates an [`Environment`](@ref) from an [`Instance`](@ref) of the maintenance benchmark.
"""
function Environment(instance::Instance)
    degradation_state = copy(starting_state(instance))
    return Environment(; instance, step=1, degradation_state)
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
Draw random degradations for all components using `rng`.
"""
function degrad!(env::Environment, rng::AbstractRNG)
    N = component_count(env)
    n = degradation_levels(env)
    p = degradation_probability(env)

    for i in 1:N
        if env.degradation_state[i] < n && rand(rng) < p
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
    n = degradation_levels(env)
    return failure_cost(env) * count(==(n), env.degradation_state)
end

"""
$TYPEDSIGNATURES

Resets the environment to the initial state:
- reset the step to 1
- reset the degradation state to the starting state
"""
function Utils.reset!(env::Environment, ::AbstractRNG)
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
function Utils.step!(env::Environment, maintenance::BitVector, rng::AbstractRNG)
    @assert !Utils.is_terminated(env) "Environment is terminated, cannot act!"
    cost = maintenance_cost(env, maintenance) + degradation_cost(env)
    degrad!(env, rng)
    maintain!(env, maintenance)
    env.step += 1
    return cost
end
