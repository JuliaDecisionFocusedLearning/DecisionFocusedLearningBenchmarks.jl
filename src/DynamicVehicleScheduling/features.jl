function must_dispatch_in_zone(state::DVSPState)
    (; state_instance, is_must_dispatch) = state

    startTimes = state_instance.start_time
    serviceTimes = state_instance.service_time
    durations = state_instance.duration

    n = length(startTimes)
    must_dispatch_counts = zeros(n)

    # For each customer j
    for j in 1:n
        # Count how many must-dispatch customers i can reach j
        for i in 2:n
            if is_must_dispatch[i] && i != j
                # Check if customer i can reach customer j in time
                if startTimes[i] + serviceTimes[i] + durations[i, j] < startTimes[j]
                    must_dispatch_counts[j] += 1
                end
            end
        end
    end

    return must_dispatch_counts
end

function count_reachable_from(state::DVSPState)
    (; state_instance) = state

    startTimes = state_instance.start_time
    serviceTimes = state_instance.service_time
    durations = state_instance.duration

    n = length(startTimes)
    reachable_counts = zeros(n)

    # For each customer j
    for j in 1:n
        # Count how many customers i are reachable from j
        for i in 2:n
            if i != j
                # Check if customer i can reach customer j in time
                if startTimes[j] + serviceTimes[j] + durations[j, i] < startTimes[i]
                    reachable_counts[j] += 1
                end
            end
        end
    end

    return reachable_counts
end

function count_reachable_to(state::DVSPState)
    (; state_instance) = state

    startTimes = state_instance.start_time
    serviceTimes = state_instance.service_time
    durations = state_instance.duration

    n = length(startTimes)
    reachable_counts = zeros(n)

    # For each customer j
    for j in 1:n
        # Count how many customers i can reach j
        for i in 2:n
            if i != j
                # Check if customer i can reach customer j in time
                if startTimes[i] + serviceTimes[i] + durations[i, j] < startTimes[j]
                    reachable_counts[j] += 1
                end
            end
        end
    end

    return reachable_counts
end

function quantile_reachable_new_requests(
    state::DVSPState,
    instance::Instance;
    n_samples::Int=100,
    quantiles=[i * 0.1 for i in 1:9],
)
    (; state_instance, current_epoch) = state
    (; static_instance, epoch_duration, Δ_dispatch, max_requests_per_epoch) = instance

    startTimes = state_instance.start_time
    serviceTimes = state_instance.service_time
    durations = state_instance.duration
    n_current = length(startTimes)

    # Time window for next epoch
    next_time = epoch_duration * current_epoch + Δ_dispatch
    min_time = minimum(static_instance.start_time)
    max_time = maximum(static_instance.start_time)
    N = customer_count(static_instance)

    # Store reachability percentages for each customer across samples
    reachability_matrix = zeros(Float64, n_current, n_samples)

    rng = MersenneTwister(42)
    for s in 1:n_samples
        # Sample new requests similar to scenario generation
        coordinate_indices = sample_indices(rng, max_requests_per_epoch, N)
        sampled_start_times = sample_times(
            rng, max_requests_per_epoch, max(min_time, next_time), max_time
        )
        service_time_indices = sample_indices(rng, max_requests_per_epoch, N)

        # Check feasibility (can reach from depot)
        depot_durations = static_instance.duration[1, coordinate_indices]
        is_feasible = next_time .+ depot_durations .<= sampled_start_times

        feasible_coords = coordinate_indices[is_feasible]
        feasible_start_times = sampled_start_times[is_feasible]
        feasible_service_times = static_instance.service_time[service_time_indices[is_feasible]]

        n_new = length(feasible_coords)
        if n_new == 0
            continue  # No reachable requests in this sample
        end

        # For each current customer, count how many new requests it can reach
        for j in 1:n_current
            reachable_count = 0
            for k in 1:n_new
                # Get duration from current customer location to new request location
                customer_loc = state.location_indices[j]
                new_loc = feasible_coords[k]
                travel_time = static_instance.duration[customer_loc, new_loc]
                travel_time_back = static_instance.duration[new_loc, customer_loc]

                # Check if customer j can reach new request k or if k can reach j
                if startTimes[j] + serviceTimes[j] + travel_time <
                   feasible_start_times[k] ||
                    startTimes[j] >
                   feasible_start_times[k] + feasible_service_times[k] + travel_time_back
                    reachable_count += 1
                end
            end
            reachability_matrix[j, s] = reachable_count / n_new
        end
    end

    # Compute quantiles for each customer
    quantile_features = zeros(Float64, n_current, length(quantiles))
    for j in 1:n_current
        quantile_features[j, :] = quantile(reachability_matrix[j, :], quantiles)
    end

    return quantile_features
end

function get_features_quantileTimeToRequests(state::DVSPState, instance::Instance)
    quantiles = [i * 0.1 for i in 1:9]
    a = instance.static_instance.duration[state.location_indices, 2:end]
    quantileTimeToRequests = mapslices(x -> quantile(x, quantiles), a; dims=2)
    return quantileTimeToRequests
end

function compute_model_free_features(state::DVSPState, instance::Instance)
    (; state_instance, is_postponable, is_must_dispatch) = state

    startTimes = state_instance.start_time
    endTimes = startTimes .+ state_instance.service_time
    timeDepotRequest = state_instance.duration[:, 1]
    timeRequestDepot = state_instance.duration[1, :]

    slack_next_epoch = startTimes .- instance.epoch_duration

    must_dispatch_counts = must_dispatch_in_zone(state)
    nb_must_dispatch = sum(is_must_dispatch)
    if nb_must_dispatch > 0
        must_dispatch_counts ./= nb_must_dispatch
    end

    reachable_to_ratios = count_reachable_to(state) ./ (length(startTimes) - 1)
    reachable_from_ratios = count_reachable_from(state) ./ (length(startTimes) - 1)
    reachable_ratios = reachable_to_ratios .+ reachable_from_ratios

    model_free_features = hcat(
        startTimes[is_postponable], # 1
        endTimes[is_postponable], # 2
        timeDepotRequest[is_postponable], # 3
        timeRequestDepot[is_postponable], # 4
        slack_next_epoch[is_postponable], # 5
        must_dispatch_counts[is_postponable], # 6
        reachable_to_ratios[is_postponable], # 7
        reachable_from_ratios[is_postponable], # 8
        reachable_ratios[is_postponable], # 9
    )
    return model_free_features
end

function compute_model_aware_features(state::DVSPState, instance::Instance)
    quantileTimeToRequests = get_features_quantileTimeToRequests(state, instance)
    quantileReachableNewRequests = quantile_reachable_new_requests(state, instance)
    model_aware_features = hcat(quantileTimeToRequests, quantileReachableNewRequests)
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
