function first_stage_optimal_solution(inst::TwoStageSpanningTreeInstance, θ::AbstractMatrix; M=20.0)
	S = nb_scenarios(inst)
	E = ne(inst.graph)

	# first stage objective value
    edge_weight_vector = inst.first_stage_costs .+ vec(sum(θ; dims=2)) ./ S

    edges_index_with_negative_cost = [e for e in 1:E if edge_weight_vector[e] < 0]

    value = 0.0
    if length(edges_index_with_negative_cost) > 0
        value = sum(M * edge_weight_vector[e] for e in edges_index_with_negative_cost)
    end

    grad = zeros(E, S)
	grad[edges_index_with_negative_cost, :] .= M / S
    return value, grad
end;

function second_stage_optimal_solution!(
    instance::TwoStageSpanningTreeInstance,
    θ::AbstractMatrix,
    scenario::Int,
    grad::AbstractMatrix,
)
	(; graph, second_stage_costs) = instance
    S = nb_scenarios(instance)

	weights = min.(-θ[:, scenario], second_stage_costs[:, scenario])

    (; value, tree) = kruskal(graph, weights)

	# update gradient
	slice = (-θ[:, scenario] .< second_stage_costs[:, scenario]) .&& tree
	grad[slice, scenario] .-= 1 / S

    return value ./ S
end;

function lagrangian_function_value_gradient(inst::TwoStageSpanningTreeInstance, θ::AbstractMatrix)
    value, grad = first_stage_optimal_solution(inst, θ)

	S = nb_scenarios(inst)
    values = zeros(S)
    for s in 1:S
		# Different part of grad are modified
        values[s] = second_stage_optimal_solution!(inst, θ, s, grad)
    end
    value += sum(values)
    return value, grad
end;

function lagrangian_heuristic(θ::AbstractMatrix; inst::TwoStageSpanningTreeInstance)
    # Retrieve - y_{es} / S from θ by computing the gradient
	(; graph) = inst
	S = nb_scenarios(inst)
    grad = zeros(ne(graph), S)
    for s in 1:S
        second_stage_optimal_solution!(inst, θ, s, grad)
    end
    # Compute the average (over s) y_{es} and build a graph that is a candidate spannning tree (but not necessarily a spanning tree nor a forest)
    average_x = -vec(sum(grad; dims=2))
    weights = average_x .> 0.5
    # Build a spanning tree that contains as many edges of our candidate as possible
    _, tree_from_candidate = kruskal(graph, weights; minimize=false)
    # Keep only the edges that are in the initial candidate graph and in the spanning tree
    forest = weights .&& tree_from_candidate
    sol = solution_from_first_stage_forest(forest, inst)
	# v, _ = evaluate_first_stage_solution(inst, forest)
    return solution_value(sol, inst), forest
end;

"""
$TYPEDSIGNATURES

Return an heuristic solution using a combination of lagarngian relaxation and lagrangian heuristic.
"""
function lagrangian_relaxation(
    inst::TwoStageSpanningTreeInstance; nb_epochs=100, stop_gap=1e-8
)
    θ = zeros(ne(inst.graph), nb_scenarios(inst))

    opt = Adam()

    lb = -Inf
    ub = Inf
    best_theta = θ
    forest = Edge{Int}[]

    last_ub_epoch = -1000

    lb_history = Float64[]
    ub_history = Float64[]
    for epoch in 1:nb_epochs
        value, grad = lagrangian_function_value_gradient(inst, θ)

        if value > lb
            lb = value
            best_theta = θ
            if epoch > last_ub_epoch + 100
                last_ub_epoch = epoch
                ub, forest = lagrangian_heuristic(θ; inst=inst)
                if (ub - lb) / abs(lb) <= stop_gap
                    @info "Stopped after $epoch gap smaller than $stop_gap"
                    break
                end
            end
        end
        Flux.update!(opt, θ, -grad)
        push!(lb_history, value)
        push!(ub_history, ub)
    end

	ub, forest = lagrangian_heuristic(best_theta; inst=inst)
    solution = solution_from_first_stage_forest(forest, inst)
    return solution, (; lb, ub, best_theta, lb_history, ub_history)
end
