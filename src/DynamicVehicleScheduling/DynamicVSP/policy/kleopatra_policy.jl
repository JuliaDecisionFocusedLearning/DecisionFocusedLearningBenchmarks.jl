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
    (; prize_predictor, has_2D_features) = π
    x = has_2D_features ? compute_2D_features(env) : compute_features(env)
    θ = prize_predictor(x)
    routes = prize_collecting_vsp(θ; instance=get_state(env), model_builder)
    return routes
end

# function run_policy!(
#     π::KleopatraVSP, env::DVSPEnv; check_feasibility=true, model_builder=highs_model
# )
#     # reset environment, and initialize variables
#     reset!(env)
#     total_cost = 0
#     epoch_routes = Vector{Vector{Int}}[]

#     # epoch loop
#     while !is_terminated(env)
#         next_epoch!(env)
#         state_routes = π(env; model_builder)
#         check_feasibility && @assert is_feasible(get_state(env), state_routes)
#         env_routes = env_routes_from_state_routes(env, state_routes)
#         push!(epoch_routes, env_routes)
#         local_cost = apply_decision!(env, env_routes)
#         total_cost += local_cost
#     end

#     return total_cost, epoch_routes
# end
