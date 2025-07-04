"""
$TYPEDEF

Lazy policy for the Dynamic Vehicle Scheduling Problem.
Dispatch customers only when necessary (i.e. must-dispatch).
"""
struct LazyVSPPolicy <: AbstractDynamicVSPPolicy end

function (π::LazyVSPPolicy)(env::DVSPEnv; model_builder=highs_model)
    state = observe(env)
    nb_postponable_requests = sum(state.is_postponable)
    θ = ones(nb_postponable_requests) * -1e9
    routes = prize_collecting_vsp(θ; instance=state, model_builder)
    return routes
end
