"""
$TYPEDEF

Kleopatra policy for the Dynamic Vehicle Scheduling Problem.
"""
struct KleopatraVSPPolicy{P} <: AbstractDynamicVSPPolicy
    prize_predictor::P
    has_2D_features::Bool
end

"""
$TYPEDSIGNATURES

Custom constructor for [`KleopatraVSPPolicy`](@ref).
"""
function KleopatraVSPPolicy(prize_predictor; has_2D_features=nothing)
    has_2D_features = if isnothing(has_2D_features)
        size(prize_predictor[1].weight, 2) == 2
    else
        has_2D_features
    end
    return KleopatraVSPPolicy(prize_predictor, has_2D_features)
end

function (π::KleopatraVSPPolicy)(env::DVSPEnv; model_builder=highs_model)
    state = observe(env)
    (; prize_predictor, has_2D_features) = π
    x = has_2D_features ? compute_2D_features(env) : compute_features(env)
    θ = prize_predictor(x)
    routes = prize_collecting_vsp(θ; instance=state, model_builder)
    return routes
end
