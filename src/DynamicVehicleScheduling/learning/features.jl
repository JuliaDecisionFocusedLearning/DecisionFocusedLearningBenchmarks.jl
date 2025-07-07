function get_features_quantileTimeToRequests(state::DVSPState, instance::Instance)
    quantiles = [i * 0.1 for i in 1:9]
    a = instance.static_instance.duration[state.location_indices, 2:end]
    quantileTimeToRequests = mapslices(x -> quantile(x, quantiles), a; dims=2)
    return quantileTimeToRequests
end

function compute_model_free_features(state::DVSPState, instance::Instance)
    (; state_instance, is_postponable) = state

    startTimes = state_instance.start_time
    endTimes = startTimes .+ state_instance.service_time
    timeDepotRequest = state_instance.duration[:, 1]
    timeRequestDepot = state_instance.duration[1, :]

    slack_next_epoch = startTimes .- instance.epoch_duration

    model_free_features = hcat(
        startTimes[is_postponable], # 1
        endTimes[is_postponable], # 2
        timeDepotRequest[is_postponable], # 3
        timeRequestDepot[is_postponable], # 4
        slack_next_epoch[is_postponable], # 5-14
    )
    return model_free_features
end

function compute_model_aware_features(state::DVSPState, instance::Instance)
    quantileTimeToRequests = get_features_quantileTimeToRequests(state, instance)
    model_aware_features = quantileTimeToRequests
    return model_aware_features[state.is_postponable, :]
end

function compute_features(state::DVSPState, instance::Instance)
    model_free_features = compute_model_free_features(state, instance)
    model_aware_features = compute_model_aware_features(state, instance)
    return hcat(model_free_features, model_aware_features)'
end

function compute_features(env::DVSPEnv)
    return compute_features(env.state, env.instance)
end
