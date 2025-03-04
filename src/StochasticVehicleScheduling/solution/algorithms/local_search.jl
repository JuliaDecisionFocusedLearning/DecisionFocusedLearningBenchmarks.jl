"""
$TYPEDSIGNATURES

Return the optimal solution of the deterministic VSP problem associated to `instance`.
The objective function is `vehicle_cost * nb_vehicles + include_delays * delay_cost * sum_of_travel_times`
Note: If you have Gurobi, use `grb_model` as `model_builder` instead od `highs_model`.
"""
function solve_deterministic_VSP(
    instance::Instance; include_delays=true, model_builder=highs_model, verbose=false
)
    (; city, graph) = instance

    travel_times = [
        distance(task1.end_point, task2.start_point) for task1 in city.tasks,
        task2 in city.tasks
    ]

    model = model_builder()
    verbose || set_silent(model)

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

    return JuMP.objective_value(model), solution
end

"""
$TYPEDSIGNATURES

Select one random (uniform) task and move it to another random (uniform) feasible vehicle
"""
function move_one_random_task!(path_value::BitMatrix, graph::AbstractGraph)
    nb_tasks = size(path_value, 2)
    selected_task = rand(DiscreteUniform(1, nb_tasks))
    selected_vehicle = find_first_one(@view path_value[:, selected_task])

    can_be_inserted = Int[]
    # do not empty if already empty
    empty_encountered = false  #sum(@view path_value[selected_vehicle, :]) == 1 ? true : false
    for i in 1:nb_tasks
        if i == selected_vehicle
            continue
        end
        # else
        is_empty = false
        if selected_task > 1
            before = @view path_value[i, 1:(selected_task - 1)]
            if any(before)
                aaa = find_first_one(reverse(before))
                @assert aaa >= 0
                precedent_task = selected_task - aaa
                if !has_edge(graph, precedent_task + 1, selected_task + 1)
                    continue
                end
            elseif empty_encountered
                continue
            else # if !empty_encountered
                is_empty = true
            end
        end

        if selected_task < nb_tasks
            after = @view path_value[i, (selected_task + 1):end]
            if any(after)
                bbb = find_first_one(@view path_value[i, (selected_task + 1):end])
                @assert bbb >= 0
                next_task = selected_task + bbb

                if !has_edge(graph, selected_task + 1, next_task + 1)
                    continue
                end
            elseif empty_encountered
                continue
            elseif !empty_encountered && is_empty
                empty_encountered = true
            end
        end

        push!(can_be_inserted, i)
    end
    if length(can_be_inserted) == 0
        @warn "No space to be inserted" selected_task path_value
        return nothing
    end
    new_vehicle = rand(can_be_inserted)
    path_value[selected_vehicle, selected_task] = false
    path_value[new_vehicle, selected_task] = true
    return nothing
end

"""
$TYPEDSIGNATURES

Very simple local search heuristic, using the neighborhood defined by `move_one_random_task`
"""
function _local_search(solution::Solution, instance::Instance; nb_it::Integer=100)
    best_solution = copy(solution.path_value)
    best_value = evaluate_solution(solution, instance)
    history_x = [0]
    history_y = [best_value]

    candidate_solution = copy(solution.path_value)
    for it in 1:nb_it
        move_one_random_task!(candidate_solution, instance.graph)

        value = evaluate_solution(candidate_solution, instance)
        if value <= best_value # keep changes
            best_solution = copy(candidate_solution)
            best_value = value
            push!(history_x, it)
            push!(history_y, best_value)
        else # revert changes
            candidate_solution = copy(best_solution)
        end
    end
    return Solution(best_solution, instance), best_value, history_x, history_y
end

"""
$TYPEDSIGNATURES

Very simple heuristic, using [`local_search`](@ref)
    initialised with the solution of the deterministic Linear program
"""
function local_search(instance::Instance; num_iterations=1000)
    _, initial_solution = solve_deterministic_VSP(instance)
    sol, _, _, _ = _local_search(initial_solution, instance; nb_it=num_iterations)
    return sol
end
