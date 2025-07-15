"""
$TYPEDEF

Greedy policy for the Dynamic Vehicle Scheduling Problem.
Dispatch customers as soon as they appear.
"""
struct GreedyVSPPolicy <: AbstractDynamicVSPPolicy end

function (π::GreedyVSPPolicy)(env::DVSPEnv; model_builder=highs_model)
    state = observe(env)
    (; is_postponable) = state
    nb_postponable_requests = sum(is_postponable)
    θ = ones(nb_postponable_requests) * 1e9
    routes = prize_collecting_vsp(θ; instance=state, model_builder)
    return routes
end
