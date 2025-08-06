function expert_policy(env::Environment)
    N = item_count(env)
    K = assortment_size(env)
    best_S = falses(N)
    best_revenue = -1.0
    S_vec = falses(N)
    for S in combinations(1:N, K)
        S_vec .= false
        S_vec[S] .= true
        expected_revenue = compute_expected_revenue(env, S_vec)
        if expected_revenue > best_revenue
            best_S, best_revenue = copy(S_vec), expected_revenue
        end
    end
    return best_S
end

function greedy_policy(env::Environment)
    maximizer = generate_maximizer(env.instance.config)
    return maximizer(prices(env))
end

function run_policy(env::Environment, episodes::Int; first_seed=1, policy=expert_policy)
    dataset = []
    rev_global = Float64[]
    for i in 1:episodes
        rev_episode = 0.0
        CommonRLInterface.reset!(env; seed=first_seed - 1 + i, reset_seed=true)
        training_instances = []
        while !CommonRLInterface.terminated(env)
            S = policy(env)
            features = CommonRLInterface.observe(env)
            push!(training_instances, DataSample(; x=features, y_true=S))
            reward = CommonRLInterface.act!(env, S)
            rev_episode += reward
        end
        push!(rev_global, rev_episode)
        push!(dataset, training_instances)
    end
    return mean(rev_global), rev_global, dataset
end
