
"""
$TYPEDSIGNATURES

Greedy policy that maintains components when they are in the last state before failure, up to the maintenance capacity.
"""
function greedy_policy(env::Environment)
    maximizer = generate_maximizer(env.instance.config)
    return maximizer(prices(env)[1:item_count(env)])
end
