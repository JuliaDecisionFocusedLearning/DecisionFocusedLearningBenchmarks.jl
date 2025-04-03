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
    π::AbstractDynamicVSPPolicy, env::DVSPEnv; check_feasibility=true, kwargs...
)
    # reset environment, and initialize variables
    reset!(env)
    total_cost = 0
    epoch_routes = Vector{Vector{Int}}[]

    # epoch loop
    while !is_terminated(env)
        next_epoch!(env)
        state_routes = π(env; kwargs...)
        check_feasibility && @assert is_feasible(get_state(env), state_routes)
        env_routes = env_routes_from_state_routes(env, state_routes)
        push!(epoch_routes, env_routes)
        local_cost = apply_decision!(env, env_routes)
        total_cost += local_cost
    end

    return total_cost, epoch_routes
end
