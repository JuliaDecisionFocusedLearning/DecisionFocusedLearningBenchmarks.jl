"""
    delay_sum(path, slacks, delays)

Evaluate the total delay along path.
"""
function delay_sum(path, slacks, delays)
    nb_scenarios = size(delays, 2)
    old_v = path[1]
    R = delays[old_v, :]
    C = 0.0
    for v in path[2:(end - 1)]
        @. R = max(R - slacks[old_v, v], 0) + delays[v, :]
        C += sum(R) / nb_scenarios
        old_v = v
    end
    return C
end

"""
    column_generation(instance::Instance)

Note: If you have Gurobi, use `grb_model` as `model_builder` instead of `glpk_model`.
"""
function column_generation(
    instance::Instance;
    model_builder=highs_model,
    bounding,
    use_convex_resources,
    scenario_range=nothing,
    silent=true,
)
    (; graph, vehicle_cost, delay_cost) = instance

    Ω = isnothing(scenario_range) ? (1:get_nb_scenarios(instance)) : scenario_range
    intrinsic_delays = instance.intrinsic_delays[:, Ω]
    slacks = deepcopy(instance.slacks)
    for edge in edges(graph)
        slacks[src(edge), dst(edge)] = slacks[src(edge), dst(edge)][Ω]
    end

    nb_nodes = nv(graph)
    job_indices = 2:(nb_nodes - 1)

    model = model_builder()
    set_silent(model)

    @variable(model, λ[v in 1:nb_nodes])

    @objective(model, Max, sum(λ[v] for v in job_indices))

    initial_paths = [[1, v, nb_nodes] for v in job_indices]
    @constraint(
        model,
        con[p in initial_paths],
        delay_cost * delay_sum(p, slacks, intrinsic_delays) + vehicle_cost -
        sum(λ[v] for v in job_indices if v in p) >= 0
    )
    @constraint(model, λ[1] == 0)
    @constraint(model, λ[nb_nodes] == 0)

    new_paths = Vector{Int}[]
    cons = []

    while true
        optimize!(model)
        λ_val = value.(λ)
        (; c_star, p_star) = stochastic_routing_shortest_path(
            graph,
            slacks,
            intrinsic_delays,
            λ_val ./ delay_cost;
            bounding,
            use_convex_resources,
        )
        λ_sum = sum(λ_val[v] for v in job_indices if v in p_star)
        path_cost = delay_cost * c_star + λ_sum + vehicle_cost
        silent || @info path_cost - λ_sum
        if path_cost - λ_sum > -1e-10
            break
        end
        push!(new_paths, p_star)
        push!(
            cons,
            @constraint(
                model, path_cost - sum(λ[v] for v in job_indices if v in p_star) >= 0
            )
        )
    end

    c_low = JuMP.objective_value(model)
    columns = unique(cat(initial_paths, new_paths; dims=1))
    return columns, c_low::Float64, value.(λ)
end

"""
$TYPEDSIGNATURES

Note: If you have Gurobi, use `grb_model` as `model_builder` instead od `glpk_model`.
"""
function compute_solution_from_selected_columns(
    instance::Instance,
    paths;
    bin=true,
    model_builder=highs_model,
    scenario_range=nothing,
    silent=true,
)
    (; graph, vehicle_cost, delay_cost) = instance

    nb_nodes = nv(graph)
    job_indices = 2:(nb_nodes - 1)

    Ω = isnothing(scenario_range) ? (1:get_nb_scenarios(instance)) : scenario_range
    intrinsic_delays = instance.intrinsic_delays[:, Ω]
    slacks = deepcopy(instance.slacks)
    for edge in edges(graph)
        slacks[src(edge), dst(edge)] = slacks[src(edge), dst(edge)][Ω]
    end

    model = model_builder()
    silent && set_silent(model)

    if bin
        @variable(model, y[p in paths], Bin)
    else
        @variable(model, y[p in paths] >= 0)
    end

    @objective(
        model,
        Min,
        sum(
            (delay_cost * delay_sum(p, slacks, intrinsic_delays) + vehicle_cost) * y[p] for
            p in paths
        )
    )

    @constraint(model, con[v in job_indices], sum(y[p] for p in paths if v in p) == 1)

    optimize!(model)

    sol = value.(y)
    return JuMP.objective_value(model)::Float64,
    sol,
    paths[isapprox.([sol[p] for p in paths], 1.0)]
end

"""
$TYPEDSIGNATURES

Solve input instance using column generation.
"""
function column_generation_algorithm(
    instance::Instance;
    scenario_range=nothing,
    bounding=true,
    use_convex_resources=true,
    silent=true,
    close_gap=false,
)
    Ω = isnothing(scenario_range) ? (1:get_nb_scenarios(instance)) : scenario_range
    columns, c_low, λ_val = column_generation(
        instance;
        bounding=bounding,
        use_convex_resources=use_convex_resources,
        scenario_range=Ω,
        silent=silent,
    )
    c_upp, _, sol = compute_solution_from_selected_columns(
        instance, columns; scenario_range=Ω, silent=silent
    )

    if close_gap && abs(c_upp - c_low) > 1e-6
        (; vehicle_cost, delay_cost, graph, slacks) = instance
        intrinsic_delays = instance.intrinsic_delays[:, Ω]
        slacks = deepcopy(instance.slacks)
        for edge in edges(graph)
            slacks[src(edge), dst(edge)] = slacks[src(edge), dst(edge)][Ω]
        end
        threshold = (c_upp - c_low - vehicle_cost) / delay_cost
        additional_paths, _ = stochastic_routing_shortest_path_with_threshold(
            graph, slacks, intrinsic_delays, λ_val ./ delay_cost; threshold
        )
        columns = unique(cat(columns, additional_paths; dims=1))

        _, _, sol = compute_solution_from_selected_columns(
            instance, columns; scenario_range=Ω, silent=silent
        )
    end

    col_solution = solution_from_paths(sol, instance)
    return col_solution.value
end
