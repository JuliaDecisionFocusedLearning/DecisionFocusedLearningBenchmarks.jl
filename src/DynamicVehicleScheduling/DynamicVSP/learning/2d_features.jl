function get_features_meanTimeToRequests(env::DVSPEnv)
    quantiles = [0.5]
    a = env.config.static_instance.duration[
        env.customer_index[.!env.request_is_dispatched], 2:end
    ]
    quantileTimeToRequests = mapslices(x -> quantile(x, quantiles), a; dims=2)
    return quantileTimeToRequests
end

function compute_2D_features(env::DVSPEnv)
    state = env.state
    timeDepotRequest = state.instance.duration[:, 1][state.is_postponable]
    quantileTimeToRequests = get_features_meanTimeToRequests(env)[state.is_postponable]
    return hcat(timeDepotRequest, quantileTimeToRequests)'
end
