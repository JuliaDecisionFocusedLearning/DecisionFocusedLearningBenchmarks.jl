function get_features_quantileTimeToRequests(env::DVSPEnv)
    quantiles = [i * 0.1 for i in 1:9]
    a = env.config.static_instance.duration[
        env.customer_index[.!env.request_is_dispatched], 2:end
    ]
    quantileTimeToRequests = mapslices(x -> quantile(x, quantiles), a; dims=2)
    return quantileTimeToRequests
end

function compute_model_free_features(state::VSPState; env::DVSPEnv)
    (; instance, is_postponable) = state

    startTimes = instance.start_time
    endTimes = startTimes .+ instance.service_time
    timeDepotRequest = instance.duration[:, 1]
    timeRequestDepot = instance.duration[1, :]

    slack_next_epoch = startTimes .- env.config.epoch_duration

    model_free_features = hcat(
        startTimes[is_postponable], # 1
        endTimes[is_postponable], # 2
        timeDepotRequest[is_postponable], # 3
        timeRequestDepot[is_postponable], # 4
        slack_next_epoch[is_postponable], # 5-14
    )
    return model_free_features
end

function compute_model_aware_features(state::VSPState; env::DVSPEnv)
    quantileTimeToRequests = get_features_quantileTimeToRequests(env)
    model_aware_features = quantileTimeToRequests
    return model_aware_features[state.is_postponable, :]
end

function compute_features(env::DVSPEnv)
    state = env.state
    model_free_features = compute_model_free_features(state; env)
    model_aware_features = compute_model_aware_features(state; env)
    return hcat(model_free_features, model_aware_features)'
end

# ? why is this needed
function model_free_features_critic(state::VSPState; env::DVSPEnv)
    (; instance) = state
    startTimes = instance.start_time
    endTimes = instance.service_time .+ instance.start_time
    timeDepotRequest = instance.duration[:, 1]
    timeRequestDepot = instance.duration[1, :]
    slack_next_epoch = startTimes .- env.config.epoch_duration
    model_free_features = hcat(
        startTimes, endTimes, timeDepotRequest, timeRequestDepot, slack_next_epoch
    )
    return model_free_features
end

# ?
function compute_critic_features(env::DVSPEnv)
    state = env.state
    model_free_features = model_free_features_critic(state; env)
    model_aware_features = get_features_quantileTimeToRequests(env)
    postpon = state.is_postponable
    return hcat(model_free_features, model_aware_features, postpon)'
end

# ?
function compute_critic_2D_features(env::DVSPEnv)
    state = env.state
    timeDepotRequest = state.instance.duration[:, 1]
    quantileTimeToRequests = get_features_meanTimeToRequests(env)
    postpon = state.is_postponable
    # time_postpon = timeDepotRequest .* postpon
    # quant_postpon = quantileTimeToRequests .* postpon
    return hcat(timeDepotRequest, quantileTimeToRequests, postpon)'
end
