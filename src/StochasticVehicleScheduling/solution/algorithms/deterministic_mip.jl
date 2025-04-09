"""
$TYPEDSIGNATURES

Solves the deterministic version of the vehicle scheduling problem using a MIP model.
Does not take into account the stochastic nature of the problem.
"""
function deterministic_mip(instance::Instance; model_builder=highs_model, silent=true)
    (; graph, vehicle_cost) = instance
    nb_nodes = nv(graph)
    job_indices = 2:(nb_nodes - 1)
    nodes = 1:nb_nodes

    # Model definition
    model = model_builder()
    silent && set_silent(model)

    # Variables and objective function
    @variable(model, y[u in nodes, v in nodes; has_edge(graph, u, v)], Bin)

    @objective(
        model,
        Min,
        vehicle_cost * sum(y[1, v] for v in job_indices) # nb_vehicles
    )

    # Flow contraints
    @constraint(
        model,
        flow[i in job_indices],
        sum(y[j, i] for j in inneighbors(graph, i)) ==
            sum(y[i, j] for j in outneighbors(graph, i))
    )
    @constraint(
        model,
        unit_demand[i in job_indices],
        sum(y[j, i] for j in inneighbors(graph, i)) == 1
    )

    # Solve model
    optimize!(model)
    solution = value.(y)

    sol = solution_from_JuMP_array(solution, graph)
    return sol
end
