"""
    move_one_random_task!(path_value::BitMatrix, graph::AbstractGraph)

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
                precedent_task = selected_task - find_first_one(reverse(before))
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
                next_task =
                    selected_task +
                    find_first_one(@view path_value[i, (selected_task + 1):end])
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
    local_search(solution::Solution, instance::AbstractInstance; nb_it::Integer=100)

Very simple local search heuristic, using the neighborhood defined by `move_one_random_task`
"""
function local_search(solution::Solution, instance::AbstractInstance; nb_it::Integer=100)
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
    heuristic_solution(instance::AbstractInstance; nb_it=100)

Very simple heuristic, using [`local_search`](@ref)
    initialised with the solution of the deterministic Linear program
"""
function heuristic_solution(instance::AbstractInstance; nb_it=100)
    _, initial_solution = solve_deterministic_VSP(instance)
    sol, _, _, _ = local_search(initial_solution, instance; nb_it=nb_it)
    return sol
end
