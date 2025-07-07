function get_features_meanTimeToRequests(state::DVSPState, instance::Instance)
    quantiles = [0.5]
    a = instance.static_instance.duration[state.location_indices, 2:end]
    quantileTimeToRequests = mapslices(x -> quantile(x, quantiles), a; dims=2)
    return quantileTimeToRequests
end

function compute_2D_features(state::DVSPState, instance::Instance)
    timeDepotRequest = state.state_instance.duration[:, 1][state.is_postponable]
    quantileTimeToRequests = get_features_meanTimeToRequests(state, instance)[state.is_postponable]
    return hcat(timeDepotRequest, quantileTimeToRequests)'
end

function compute_2D_features(env::DVSPEnv)
    return compute_2D_features(env.state, env.instance)
end
