function greedy_policy(env::Environment; model_builder=highs_model)
    _, state = observe(env)
    N = item_count(env)
    ub = ub_per_item(env)
    Θ = zeros(N + sum(ub))
    Θ[1:N] .= prices(env)
    return (replenishment_problem(Θ; state, model_builder=model_builder))
end

function lazy_policy(env::Environment)
    N = item_count(env)
    return zeros(N)
end

function random_policy(env::Environment)
    N = item_count(env)
    cons_mat = constraints_matrix(env)
    q = quotas(env)
    replenishment = zeros(Int, N)
    order_item = randperm(N)
    t = current_epoch(env)
    for item in order_item
        max_quota_item = max(
            0,
            minimum([
                q[t, c] - sum(replenishment[j] * cons_mat[c, j] for j in 1:N) for
                c in 1:nb_constraints(env.config) if cons_mat[c, item] == 1
            ]),
        )
        replenishment[item] = rand(0:max_quota_item)
    end
    return replenishment
end
