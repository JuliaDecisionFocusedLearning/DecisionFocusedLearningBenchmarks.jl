"""
    solve_deterministic_VSP(instance::Instance; include_delays=true)

Return the optimal solution of the deterministic VSP problem associated to `instance`.
The objective function is `vehicle_cost * nb_vehicles + include_delays * delay_cost * sum_of_travel_times`
Note: If you have Gurobi, use `grb_model` as `model_builder` instead od `cbc_model`.
"""
function solve_deterministic_VSP(
    instance::Instance; include_delays=true, model_builder=cbc_model
)
    (; city, graph) = instance

    travel_times = [
        distance(task1.end_point, task2.start_point) for task1 in city.tasks,
        task2 in city.tasks
    ]

    model = model_builder()

    nb_nodes = nv(graph)
    job_indices = 2:(nb_nodes - 1)

    @variable(model, x[i=1:nb_nodes, j=1:nb_nodes; has_edge(graph, i, j)], Bin)

    @objective(
        model,
        Min,
        instance.city.vehicle_cost * sum(x[1, j] for j in job_indices) +
            include_delays *
        instance.city.delay_cost *
        sum(
            travel_times[i, j] * x[i, j] for i in 1:nb_nodes for
            j in 1:nb_nodes if has_edge(graph, i, j)
        )
    )

    @constraint(
        model,
        flow[i in job_indices],
        sum(x[j, i] for j in inneighbors(graph, i)) ==
            sum(x[i, j] for j in outneighbors(graph, i))
    )
    @constraint(
        model, demand[i in job_indices], sum(x[j, i] for j in inneighbors(graph, i)) == 1
    )

    optimize!(model)

    solution = solution_from_JuMP_array(value.(x), graph)

    return objective_value(model), solution
end
