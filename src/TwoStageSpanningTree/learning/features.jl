function _edge_adjacent_indices(graph::AbstractGraph, e::AbstractEdge)
    result = Int[]
    for v in (src(e), dst(e))
        for u in neighbors(graph, v)
            push!(result, edge_index(graph, Edge(min(u, v), max(u, v))))
        end
    end
    return result
end

function _compute_mst_indicator(
    instance::TwoStageSpanningTreeInstance, scenario::Int, weight_fn
)
    weights = [weight_fn(instance, i, scenario) for i in 1:ne(instance.graph)]
    _, tree = kruskal(instance.graph, weights)
    return tree
end

function _compute_first_stage_mst(instance::TwoStageSpanningTreeInstance)
    return _compute_mst_indicator(instance, 1, (inst, i, _) -> inst.first_stage_costs[i])
end

function _compute_second_stage_mst(instance::TwoStageSpanningTreeInstance)
    S = nb_scenarios(instance)
    indicator = falses(ne(instance.graph), S)
    for s in 1:S
        indicator[:, s] .= _compute_mst_indicator(
            instance, s, (inst, i, sc) -> inst.second_stage_costs[i, sc]
        )
    end
    return indicator
end

function _compute_best_stage_mst(instance::TwoStageSpanningTreeInstance)
    (; first_stage_costs, second_stage_costs) = instance
    S = nb_scenarios(instance)
    E = ne(instance.graph)
    bfs = falses(E, S)
    bss = falses(E, S)
    for s in 1:S
        tree = _compute_mst_indicator(
            instance,
            s,
            (inst, i, sc) -> min(inst.first_stage_costs[i], inst.second_stage_costs[i, sc]),
        )
        for i in 1:E
            tree[i] || continue
            if first_stage_costs[i] <= second_stage_costs[i, s]
                bfs[i, s] = true
            else
                bss[i, s] = true
            end
        end
    end
    return bfs, bss
end

"""
$TYPEDSIGNATURES

Compute per-edge features for a [`TwoStageSpanningTreeInstance`](@ref).

Returns a `Float32` matrix of shape `(nb_features, ne)` where `nb_features = 2 + 7 * 11`.

Features are normalized by `c_max` (first-stage costs) and `d_max` (second-stage costs) so
that all values lie in `[0, 1]`. Pass `c_max=1` and `d_max=1` to skip normalization.

# Features (normalized to [0, 1])
- `first_stage_cost / c_max`
- Quantiles (0:0.1:1) of `second_stage_cost / d_max` across scenarios
- Quantiles of `best_stage_cost / max(c_max, d_max)` across scenarios
- Quantiles of `neighbors_first_stage_cost / c_max`
- Quantiles of `neighbors_second_stage_cost / d_max` across scenarios and neighbors
- `is_in_first_stage_mst × first_stage_cost / c_max`
- Quantiles of `is_in_second_stage_mst × second_stage_cost / d_max`
- Quantiles of `is_first_in_best_stage_mst × first_stage_cost / c_max`
- Quantiles of `is_second_in_best_stage_mst × second_stage_cost / d_max`
"""
function compute_features(
    instance::TwoStageSpanningTreeInstance; c_max::Real=1, d_max::Real=1
)
    (; graph, first_stage_costs, second_stage_costs) = instance
    S = nb_scenarios(instance)
    E = ne(graph)
    cd_max = max(c_max, d_max)

    quantiles_used = 0.0:0.1:1.0
    nb_quantiles = length(quantiles_used)
    nb_features = 2 + 7 * nb_quantiles

    fs_mst = _compute_first_stage_mst(instance)
    ss_mst = _compute_second_stage_mst(instance)
    bfs_mst, bss_mst = _compute_best_stage_mst(instance)

    X = zeros(Float32, nb_features, E)

    for (i, e) in enumerate(edges(graph))
        f = 0

        function add_quantiles(realizations)
            for p in quantiles_used
                f += 1
                X[f, i] = quantile(realizations, p)
            end
        end

        # first_stage_cost
        f += 1
        X[f, i] = first_stage_costs[i] / c_max

        # second_stage_cost quantiles across scenarios
        add_quantiles(second_stage_costs[i, :] ./ d_max)

        # best_stage_cost quantiles across scenarios
        add_quantiles(min.(first_stage_costs[i], second_stage_costs[i, :]) ./ cd_max)

        # neighbor first_stage_cost quantiles
        adj = _edge_adjacent_indices(graph, e)
        add_quantiles(first_stage_costs[adj] ./ c_max)

        # neighbor second_stage_cost quantiles across scenarios and neighbors
        add_quantiles([second_stage_costs[n, s] / d_max for s in 1:S for n in adj])

        # is_in_first_stage_mst × first_stage_cost
        f += 1
        X[f, i] = fs_mst[i] * first_stage_costs[i] / c_max

        # is_in_second_stage_mst × second_stage_cost quantiles
        add_quantiles(ss_mst[i, :] .* second_stage_costs[i, :] ./ d_max)

        # is_first_in_best_stage_mst × first_stage_cost quantiles
        add_quantiles(bfs_mst[i, :] .* first_stage_costs[i] ./ c_max)

        # is_second_in_best_stage_mst × second_stage_cost quantiles
        add_quantiles(bss_mst[i, :] .* second_stage_costs[i, :] ./ d_max)
    end

    return X
end
