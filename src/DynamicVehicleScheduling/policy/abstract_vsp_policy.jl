abstract type AbstractDynamicPolicy end

function (π::AbstractDynamicPolicy)(env; kwargs...)
    throw("Not implemented")
end

"""
$TYPEDEF

Abstract type for dynamic VSP policies.
"""
abstract type AbstractDynamicVSPPolicy <: AbstractDynamicPolicy end

"""
$TYPEDSIGNATURES

Apply the policy to the environment.
"""
function run_policy!(
    π::AbstractDynamicVSPPolicy,
    env::DVSPEnv,
    scenario=env.scenario;
    check_feasibility=true,
    kwargs...,
)
    # reset environment, and initialize variables
    reset!(env)
    total_cost = 0
    epoch_routes = Vector{Vector{Int}}[]

    # epoch loop
    while !terminated(env)
        state_routes = π(env; kwargs...)
        check_feasibility && @assert is_feasible(observe(env), state_routes)
        # env_routes = env_routes_from_state_routes(env, state_routes)
        push!(epoch_routes, state_routes)
        local_cost = act!(env, state_routes, scenario)
        total_cost += local_cost
    end

    return total_cost, epoch_routes
end
