mutable struct DVSPEnv{S<:DVSPState,R<:AbstractRNG,SS} <: Utils.AbstractEnvironment
    "associated instance"
    instance::Instance
    "current state"
    state::S
    "scenario the environment will use when not given a specific one"
    scenario::Scenario
    "random number generator"
    rng::R
    "seed for the environment"
    seed::SS
end

"""
$TYPEDSIGNATURES

Constructor for [`DVSPEnv`](@ref).
"""
function DVSPEnv(instance::Instance; seed=nothing)
    rng = MersenneTwister(seed)
    scenario = Utils.generate_scenario(instance; rng)
    initial_state = DVSPState(instance; scenario[1]...)
    return DVSPEnv(instance, initial_state, scenario, rng, seed)
end

currrent_epoch(env::DVSPEnv) = current_epoch(env.state)
epoch_duration(env::DVSPEnv) = epoch_duration(env.instance)
last_epoch(env::DVSPEnv) = last_epoch(env.instance)
Δ_dispatch(env::DVSPEnv) = Δ_dispatch(env.instance)

Utils.get_seed(env::DVSPEnv) = env.seed

"""
$TYPEDSIGNATURES

Get the current state of the environment.
"""
Utils.observe(env::DVSPEnv) = nothing, env.state

current_epoch(env::DVSPEnv) = current_epoch(env.state)

"""
$TYPEDSIGNATURES

Get the current time of the environment, i.e. the start time of the current_epoch.
"""
time(env::DVSPEnv) = (current_epoch(env) - 1) * epoch_duration(env)

"""
$TYPEDSIGNATURES

Get the planning start time of the environment, i.e. the time at which vehicles routes dispatched in current epoch can depart.
"""
planning_start_time(env::DVSPEnv) = time(env) + Δ_dispatch(env)

"""
$TYPEDSIGNATURES

Check if the episode is terminated, i.e. if the current epoch is the last one.
"""
Utils.is_terminated(env::DVSPEnv) = current_epoch(env) > last_epoch(env)

"""
$TYPEDSIGNATURES

Reset the environment to its initial state.
Also reset the seed if `reset_seed` is set to true.
"""
function Utils.reset!(env::DVSPEnv; seed=get_seed(env), reset_seed=false)
    if reset_seed
        Random.seed!(env.rng, seed)
    end
    env.scenario = Utils.generate_scenario(env; rng=env.rng)
    reset_state!(env.state, env.instance; env.scenario[1]...)
    return nothing
end

"""
$TYPEDSIGNATURES

Remove dispatched customers, advance time, and add new requests to the environment.
"""
function Utils.step!(env::DVSPEnv, routes, scenario=env.scenario)
    reward = -apply_routes!(env.state, routes)
    env.state.current_epoch += 1
    if !Utils.is_terminated(env)
        add_new_customers!(env.state, env.instance; scenario[current_epoch(env)]...)
    end
    return reward
end

function Utils.generate_scenario(env::DVSPEnv; kwargs...)
    return generate_scenario(env.instance; kwargs...)
end
