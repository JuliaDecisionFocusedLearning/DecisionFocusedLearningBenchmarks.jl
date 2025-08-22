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
