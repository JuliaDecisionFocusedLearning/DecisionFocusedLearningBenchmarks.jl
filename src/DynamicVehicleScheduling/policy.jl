function greedy_policy(env::DVSPEnv; model_builder=highs_model)
    _, state = observe(env)
    (; is_postponable) = state
    nb_postponable_requests = sum(is_postponable)
    θ = ones(nb_postponable_requests) * 1e9
    routes = prize_collecting_vsp(θ; instance=state, model_builder)
    return routes
end

function lazy_policy(env::DVSPEnv; model_builder=highs_model)
    _, state = observe(env)
    nb_postponable_requests = sum(state.is_postponable)
    θ = ones(nb_postponable_requests) * -1e9
    routes = prize_collecting_vsp(θ; instance=state, model_builder)
    return routes
end

"""
$TYPEDEF

Kleopatra policy for the Dynamic Vehicle Scheduling Problem.
"""
struct KleopatraVSPPolicy{P}
    prize_predictor::P
end

function (π::KleopatraVSPPolicy)(env::DVSPEnv; model_builder=highs_model)
    x, state = observe(env)
    (; prize_predictor) = π
    # x = has_2D_features ? compute_2D_features(env) : compute_features(env)
    θ = prize_predictor(x)
    routes = prize_collecting_vsp(θ; instance=state, model_builder)
    return routes
end
