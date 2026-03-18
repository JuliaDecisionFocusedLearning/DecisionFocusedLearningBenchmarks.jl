"""
$TYPEDEF

# Fields
$TYPEDFIELDS
"""
struct TwoStageSpanningTreeSolution
    y::BitVector
    z::BitMatrix
end

"""
$TYPEDSIGNATURES

Compute the objective value of given solution for given instance.
"""
function solution_value(solution::TwoStageSpanningTreeSolution, instance::TwoStageSpanningTreeInstance)
    return dot(solution.y, instance.first_stage_costs) + dot(solution.z, instance.second_stage_costs) / nb_scenarios(instance)
end

"""
$TYPEDSIGNATURES

Check if a given solution is feasible for given instance.
"""
function is_feasible(solution::TwoStageSpanningTreeSolution, instance::TwoStageSpanningTreeInstance; verbose=true)
    (; y, z) = solution
    (; graph) = instance

    # Check that no edge was selected in both stages
    if any(y .+ z .> 1)
        verbose && @warn "Same edge selected in both stages"
        return false
    end

    # Check that each scenario is a spanning tree
    S = nb_scenarios(instance)
    for s in 1:S
        if !is_spanning_tree(y .|| z[:, s], graph)
            verbose && @warn "Scenario $s is not a spanning tree: $y, $(z[:, s]), $instance"
            return false
        end
    end

    return true
end

"""
$TYPEDSIGNATURES

Return the associated two-stage solution from given first stage forest and instance.
"""
function solution_from_first_stage_forest(forest::BitVector, instance::TwoStageSpanningTreeInstance)
	(; graph, second_stage_costs) = instance

	S = nb_scenarios(instance)
	forests = falses(ne(graph), S)
    for s in 1:S
		weights = deepcopy(second_stage_costs[:, s])
        m = minimum(weights) - 1
        m = min(0, m - 1)
        weights[forest] .= m  # set weights over forest as the minimum

		# find min spanning tree including forest
        _, tree_s = kruskal(graph, weights)
		forest_s = tree_s .- forest
		forests[:, s] .= forest_s
    end

    return TwoStageSpanningTreeSolution(forest, forests)
end
