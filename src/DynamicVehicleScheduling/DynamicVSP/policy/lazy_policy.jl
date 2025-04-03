"""
$TYPEDEF

Lazy policy for the Dynamic Vehicle Scheduling Problem.
Dispatch customers only when necessary (i.e. must-dispatch).
"""
struct LazyVSPPolicy <: AbstractDynamicVSPPolicy end

function (π::LazyVSPPolicy)(env::DVSPEnv; model_builder=highs_model)
    nb_postponable_requests = sum(get_state(env).is_postponable)
    θ = ones(nb_postponable_requests) * -1e9
    routes = prize_collecting_vsp(θ; instance=get_state(env), model_builder)
    return routes
end

# function run_policy!(π::LazyVSPPolicy, env::DVSPEnv; check_feasibility=true, kwargs...)
#     # reset environment, and initialize variables
#     reset!(env)
#     total_cost = 0
#     epoch_routes = Vector{Vector{Int}}[]

#     # epoch loop
#     while !is_terminated(env)
#         next_epoch!(env)
#         state_routes = π(env; kwargs...)
#         check_feasibility && @assert is_feasible(get_state(env), state_routes)
#         env_routes = env_routes_from_state_routes(env, state_routes)
#         push!(epoch_routes, env_routes)
#         local_cost = apply_decision!(env, env_routes)
#         total_cost += local_cost
#     end

#     return total_cost, epoch_routes
# end
