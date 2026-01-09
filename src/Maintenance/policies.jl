
"""
$TYPEDSIGNATURES

Greedy policy that maintains components when they are in the last state before failure, up to the maintenance capacity.
"""
function greedy_policy(env::Environment)
    state = env.degradation_state
    N = component_count(env)
    K = maintenance_capacity(env)
    res = falses(N)
    n = degradation_levels(env)

    idx_max = findall(==(n), state)
    take = first(idx_max, min(K, length(idx_max)))
    res[take] .= true
    remaining = K - length(take)

    if remaining > 0
        idx_second = findall(==(n - 1), state)
        take2 = first(idx_second, min(remaining, length(idx_second)))
        res[take2] .= true
    end

    return res
end
