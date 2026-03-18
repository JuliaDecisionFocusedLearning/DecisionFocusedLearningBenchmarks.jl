
function first_stage_cost(
    inst::TwoStageSpanningTreeInstance, e::AbstractEdge, scenario::Int
)
    return inst.first_stage_weights_matrix[src(e), dst(e)]
end

function scenario_second_stage_cost(
    inst::TwoStageSpanningTreeInstance, e::AbstractEdge, scenario::Int
)
    return inst.second_stage_weights[scenario][src(e), dst(e)]
end

function scenario_best_stage_cost(
    inst::TwoStageSpanningTreeInstance, e::AbstractEdge, scenario::Int
)
    return min(
        inst.second_stage_weights[scenario][src(e), dst(e)],
        inst.first_stage_weights_matrix[src(e), dst(e)],
    )
end

function edge_neighbors(g::AbstractGraph, e::AbstractEdge)
    result = Vector{AbstractEdge}()
    for v in [src(e), dst(e)]
        for u in neighbors(g, v)
            push!(result, Edge(min(u, v), max(u, v)))
        end
    end
    return result
end

function compute_minimum_spanning_tree(
    inst::TwoStageSpanningTreeInstance, scenario::Int, weight_function
)
    weights_vec = zeros(ne(inst.g))
    for e in edges(inst.g)
        weights_vec[edge_index(inst, e)] = weight_function(inst, e, scenario)
    end
    weights = get_weight_matrix_from_weight_vector(inst.g, inst.edge_index, weights_vec)
    return kruskal_mst(inst.g, weights)
end

function compute_first_stage_mst(inst::TwoStageSpanningTreeInstance)
    tree = compute_minimum_spanning_tree(inst, -1, first_stage_cost)
    fs_mst_indicator = zeros(ne(inst.g))
    for e in tree
        fs_mst_indicator[edge_index(inst, e)] = 1.0
    end
    return fs_mst_indicator
end

function compute_second_stage_mst(inst::TwoStageSpanningTreeInstance)
    ss_mst_indicator = zeros(ne(inst.g), inst.nb_scenarios)
    for scenario in 1:(inst.nb_scenarios)
        tree = compute_minimum_spanning_tree(inst, scenario, scenario_second_stage_cost)
        for e in tree
            ss_mst_indicator[edge_index(inst, e), scenario] = 1.0
        end
    end
    return ss_mst_indicator
end

function compute_best_stage_mst(inst::TwoStageSpanningTreeInstance)
    bfs_mst_indicator = zeros(ne(inst.g), inst.nb_scenarios)
    bss_mst_indicator = zeros(ne(inst.g), inst.nb_scenarios)
    for scenario in 1:(inst.nb_scenarios)
        tree = compute_minimum_spanning_tree(inst, scenario, scenario_best_stage_cost)
        for e in tree
            e_ind = edge_index(inst, e)
            if inst.first_stage_weights_matrix[src(e), dst(e)] <
                inst.second_stage_weights[scenario][src(e), dst(e)]
                bfs_mst_indicator[e_ind, scenario] = 1.0
            else
                bss_mst_indicator[e_ind, scenario] = 1.0
            end
        end
    end
    return bfs_mst_indicator, bss_mst_indicator
end

function pivot_instance_second_stage_costs(inst::TwoStageSpanningTreeInstance)
    edgeWeights = zeros(ne(inst.g), inst.nb_scenarios)
    for s in 1:(inst.nb_scenarios)
        for e in edges(inst.g)
            edgeWeights[edge_index(inst, e), s] = inst.second_stage_weights[s][
                src(e), dst(e)
            ]
        end
    end
    return edgeWeights
end

function compute_edge_neighbors(inst::TwoStageSpanningTreeInstance)
    neighbors = Vector{Vector{Int}}(undef, ne(inst.g))
    for e in edges(inst.g)
        e_ind = edge_index(inst, e)
        neighbors_list = edge_neighbors(inst.g, e)
        neighbors[e_ind] = Vector{Int}(undef, length(neighbors_list))
        count = 0
        for f in neighbors_list
            count += 1
            neighbors[e_ind][count] = edge_index(inst, f)
        end
    end
    return neighbors
end
