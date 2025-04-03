"""
$TYPEDEF

Anticipative policy for the Dynamic Vehicle Scheduling Problem.
"""
struct AnticipativeVSPPolicy <: AbstractDynamicPolicy end

"""
$TYPEDSIGNATURES

Apply the anticipative policy to the environment.
"""
function run_policy!(::AnticipativeVSPPolicy, env::DVSPEnv; model_builder=highs_model)
    routes_anticipative = anticipative_solver(env; model_builder)
    duration = env.config.static_instance.duration[env.customer_index, env.customer_index]
    anticipative_costs = [cost(routes, duration) for routes in routes_anticipative]
    return sum(anticipative_costs), routes_anticipative
end
