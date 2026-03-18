"""
$TYPEDSIGNATURES

Compute an anticipative solution for given scenario.
"""
function anticipative_solution(instance::TwoStageSpanningTreeInstance, scenario::Int=1)
    (; graph, first_stage_costs, second_stage_costs) = instance
    scenario_second_stage_costs = @view second_stage_costs[:, scenario]
    weights = min.(first_stage_costs, scenario_second_stage_costs)
    (; value, tree) = kruskal(graph, weights)

    slice = first_stage_costs .<= scenario_second_stage_costs
    y = tree[slice]
    z = tree[.!slice]
    return (; value, y, z)
end
