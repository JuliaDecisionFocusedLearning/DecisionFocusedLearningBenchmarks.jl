"""
$TYPEDSIGNATURES

Solves the linear relaxation using a column generation algorithm.
"""
function column_generation(instance; MILP_solver=GLPK.Optimizer, tol=1e-6, verbose=true)
	(; graph, first_stage_costs, second_stage_costs) = instance

	V = nv(graph)
	E = ne(graph)
	S = nb_scenarios(instance)
	
	model = Model(MILP_solver)

	@variable(model, dummy, Bin) # dummy binary variable to activate callbacks

	@variable(model, ν[s in 1:S])
	@variable(model, μ[e in 1:E, s in 1:S])

	@objective(model, Max, sum(ν[s] for s in 1:S))

	@constraint(
		model, [e in 1:E, s in 1:S],
		second_stage_costs[e, s] / S - μ[e, s] >= 0
	)

	@constraint(
		model, [e in 1:E],
		first_stage_costs[e] - sum(μ[e, s] for s in 1:S) >= 0
	)

	trees = BitVector[]

	for s in 1:S
		_, dummy_tree = kruskal(graph, min.(first_stage_costs, second_stage_costs[:, s]))
		@constraint(
        	model,
        	ν[s] <= sum(μ[e, s] for e in 1:E if dummy_tree[e])
    	)
		push!(trees, dummy_tree)
	end

	callback_counter = 0
	function my_callback_function(cb_data)
		callback_counter += 1

        ν_val = callback_value.(cb_data, ν)
        μ_val = callback_value.(cb_data, μ)

        for s in 1:S
            val, T = kruskal(graph, @view μ_val[:, s])

			if val + tol < ν_val[s]
				push!(trees, T)
                new_constraint = @build_constraint(
                    ν[s] <= sum(μ[e, s] for e in 1:E if T[e])
                )
                MOI.submit(
                    model, MOI.LazyConstraint(cb_data), new_constraint
                )
            end
        end
    end

	set_attribute(model, MOI.LazyConstraintCallback(), my_callback_function)

	optimize!(model)
	verbose && @info "Optimal solution found after $callback_counter cuts"
	return (; value=objective_value(model), ν=value.(ν), μ=value.(μ), columns=trees)
end

function column_heuristic(instance, columns; MILP_solver=GLPK.Optimizer)
	(; graph, first_stage_costs, second_stage_costs) = instance
	E = ne(graph)
	S = nb_scenarios(instance)
	T = length(columns)

	model = Model(MILP_solver)

	@variable(model, y[e in 1:E], Bin)
	@variable(model, z[e in 1:E, s in 1:S], Bin)

	@variable(model, λ[t in 1:T, s in 1:S], Bin)

	@objective(
		model, Min,
		sum(first_stage_costs[e] * y[e] for e in 1:E) + sum(second_stage_costs[e, s] * z[e, s] for e in 1:E for s in 1:S) / S
	)

	@constraint(model, [s in 1:S], sum(λ[t, s] for t in 1:T) == 1)
	@constraint(
		model, [e in 1:E, s in 1:S],
		y[e] + z[e, s] == sum(λ[t, s] for t in 1:T if columns[t][e])
	)

	optimize!(model)

	return TwoStageSpanningTreeSolution(value.(y) .> 0.5, value.(z) .> 0.5)
end

"""
$TYPEDSIGNATURES

Column generation heuristic, that solves the linear relaxation and then outputs the solution of the proble restricted to selected columns.
Returns an heuristic solution.
"""
function column_heuristic(instance; MILP_solver=GLPK.Optimizer, verbose=true)
	(; columns) = column_generation(instance; verbose=false, MILP_solver)
	verbose && @info "Continuous relaxation solved with $(length(columns)) columns."
	return column_heuristic(instance, columns)
end
