"""
$TYPEDSIGNATURES

Returns the optimal solution of the Stochastic VSP instance, by solving the associated compact MIP.
Quadratic constraints are linearized using Mc Cormick linearization.
Note: If you have Gurobi, use `grb_model` as `model_builder` instead of `highs_model`.
"""
function compact_linearized_mip(
    instance::Instance, scenario_range=nothing; model_builder=scip_model, silent=true
)
    (; graph, slacks, intrinsic_delays, vehicle_cost, delay_cost) = instance
    nb_nodes = nv(graph)
    job_indices = 2:(nb_nodes - 1)
    nodes = 1:nb_nodes

    # Pre-processing
    ε = intrinsic_delays
    Rmax = maximum(sum(ε; dims=1))
    nb_scenarios = size(ε, 2)
    Ω = isnothing(scenario_range) ? (1:nb_scenarios) : scenario_range

    # Model definition
    model = model_builder()
    silent && set_silent(model)

    # Variables and objective function
    @variable(model, y[u in nodes, v in nodes; has_edge(graph, u, v)], Bin)
    @variable(model, R[v in nodes, ω in Ω] >= 0) # propagated delay of job v
    @variable(model, yR[u in nodes, v in nodes, ω in Ω; has_edge(graph, u, v)] >= 0) # yR[u, v] = y[u, v] * R[u, ω]

    @objective(
        model,
        Min,
        delay_cost * sum(sum(R[v, ω] for v in job_indices) for ω in Ω) / nb_scenarios # average total delay
            +
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

    # Delay propagation constraints
    @constraint(model, [ω in Ω], R[1, ω] == ε[1, ω])
    @constraint(model, R_delay_1[v in job_indices, ω in Ω], R[v, ω] >= ε[v, ω])
    @constraint(
        model,
        R_delay_2[v in job_indices, ω in Ω],
        R[v, ω] >=
            ε[v, ω] + sum(
            yR[u, v, ω] - y[u, v] * slacks[u, v][ω] for u in nodes if has_edge(graph, u, v)
        )
    )

    # Mc Cormick linearization constraints
    @constraint(
        model,
        R_McCormick_1[u in nodes, v in nodes, ω in Ω; has_edge(graph, u, v)],
        yR[u, v, ω] >= R[u, ω] + Rmax * (y[u, v] - 1)
    )
    @constraint(
        model,
        R_McCormick_2[u in nodes, v in nodes, ω in Ω; has_edge(graph, u, v)],
        yR[u, v, ω] <= Rmax * y[u, v]
    )

    # Solve model
    optimize!(model)
    solution = value.(y)

    sol = solution_from_JuMP_array(solution, graph)
    return objective_value(model), sol
end

"""
$TYPEDSIGNATURES

Returns the optimal solution of the Stochastic VSP instance, by solving the associated compact quadratic MIP.
Note: If you have Gurobi, use `grb_model` as `model_builder` instead of `highs_model`.

!!! warning
    You need to use a solver that supports quadratic constraints to use this method.
"""
function compact_mip(
    instance::Instance, scenario_range=nothing; model_builder=scip_model, silent=true
)
    (; graph, slacks, intrinsic_delays, vehicle_cost, delay_cost) = instance
    nb_nodes = nv(graph)
    job_indices = 2:(nb_nodes - 1)
    nodes = 1:nb_nodes

    # Pre-processing
    ε = intrinsic_delays
    nb_scenarios = size(ε, 2)
    Ω = isnothing(scenario_range) ? (1:nb_scenarios) : scenario_range

    # Model definition
    model = model_builder()
    silent && set_silent(model)

    # Variables and objective function
    @variable(model, y[u in nodes, v in nodes; has_edge(graph, u, v)], Bin)
    @variable(model, R[v in nodes, ω in Ω] >= 0) # propagated delay of job v
    @variable(model, yR[u in nodes, v in nodes, ω in Ω; has_edge(graph, u, v)] >= 0) # yR[u, v] = y[u, v] * R[u, ω]

    @objective(
        model,
        Min,
        delay_cost * sum(sum(R[v, ω] for v in job_indices) for ω in Ω) / nb_scenarios # average total delay
            +
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

    # Delay propagation constraints
    @constraint(model, [ω in Ω], R[1, ω] == ε[1, ω])
    @constraint(model, R_delay_1[v in job_indices, ω in Ω], R[v, ω] >= ε[v, ω])
    @constraint(
        model,
        R_delay_2[v in job_indices, ω in Ω],
        R[v, ω] >=
            ε[v, ω] +
        sum(y[u, v] * (R[u, ω] - slacks[u, v][ω]) for u in nodes if has_edge(graph, u, v))
    )

    # Solve model
    optimize!(model)
    solution = value.(y)

    sol = solution_from_JuMP_array(solution, graph)
    return objective_value(model), sol
end
