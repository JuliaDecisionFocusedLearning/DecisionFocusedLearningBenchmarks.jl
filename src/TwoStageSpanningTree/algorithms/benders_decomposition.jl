function separate_benders_cut(instance::TwoStageSpanningTreeInstance, y, s; MILP_solver, tol=1e-5)
	(; graph, second_stage_costs) = instance

	E = ne(graph)

	columns = BitVector[]

	# Feasibility cut
	model = Model(MILP_solver)

	@variable(model, dummy, Bin)

	@variable(model, νₛ <= 1)
	@variable(model, 0 <= μₛ[e in 1:E] <= 1)

	@objective(model, Max, νₛ + sum(y[e] * μₛ[e] for e in 1:E))

	function feasibility_callback(cb_data)
		μ_val = callback_value.(cb_data, μₛ)
		ν_val = callback_value(cb_data, νₛ)

		weights = -μ_val
		val, tree = kruskal(graph, weights)

		push!(columns, tree)
		
		if val + tol < ν_val
			new_constraint = @build_constraint(
				- sum(μₛ[e] for e in 1:E if tree[e]) - νₛ >= 0
			)
			MOI.submit(
				model, MOI.LazyConstraint(cb_data), new_constraint
			)
		end
	end

	set_attribute(model, MOI.LazyConstraintCallback(), feasibility_callback)
	optimize!(model)

	if objective_value(model) > tol
		return false, value.(νₛ), value.(μₛ), objective_value(model)
	end
	
	# Else, optimality cut
	optimality_model = Model(MILP_solver)

	@variable(optimality_model, dummy, Bin)

	@variable(optimality_model, νₛ)
	@variable(optimality_model, μₛ[e in 1:E] >= 0)

	@objective(
		optimality_model, Max,
		νₛ + sum(y[e] * μₛ[e] for e in 1:E) - sum(second_stage_costs[e, s] * y[e] for e in 1:E)
	)

	for tree in columns
		@constraint(
			optimality_model,
			sum(second_stage_costs[e, s] - μₛ[e] for e in 1:E if tree[e]) >= νₛ
		)
	end

	function my_callback_function(cb_data)
		μ_val = callback_value.(cb_data, μₛ)
		ν_val = callback_value(cb_data, νₛ)

		weights = second_stage_costs[:, s] .- μ_val

		val, tree = kruskal(graph, weights)

		if val - ν_val + tol < 0
			new_constraint = @build_constraint(
				sum(second_stage_costs[e, s] - μₛ[e] for e in 1:E if tree[e]) >= νₛ
			)
			MOI.submit(
				optimality_model, MOI.LazyConstraint(cb_data), new_constraint
			)
		end
	end

	set_attribute(optimality_model, MOI.LazyConstraintCallback(), my_callback_function)

	optimize!(optimality_model)

	# If primal feasible, add an optimality cut
	@assert termination_status(optimality_model) != DUAL_INFEASIBLE
	return true, value.(νₛ), value.(μₛ), objective_value(optimality_model)
end

"""
$TYPEDSIGNATURES

Returns the optimal solution using a Benders decomposition algorithm.
"""
function benders_decomposition(
    instance::TwoStageSpanningTreeInstance;
    MILP_solver=GLPK.Optimizer,
	tol=1e-6,
	verbose=true
)
	(; graph, first_stage_costs, second_stage_costs) = instance
	E = ne(graph)
	S = nb_scenarios(instance)
	
    model = Model(MILP_solver)
    @variable(model, y[e in 1:E], Bin)
    @variable(
        model,
        θ[s in 1:S] >= sum(min(0, second_stage_costs[e, s]) for e in 1:E)
    )
    @objective(
        model,
        Min,
        sum(first_stage_costs[e] * y[e] for e in 1:E) + sum(θ[s] for s in 1:S) / S
    )

    # current_scenario = 0
    callback_counter = 0
    function benders_callback(cb_data)
        if callback_counter % 10 == 0
            verbose && @info("Benders iteration: $(callback_counter)")
        end
        callback_counter += 1

        y_val = callback_value.(cb_data, y)
		θ_val = callback_value.(cb_data, θ)

        for current_scenario in 1:S
            optimality_cut, ν_val, μ_val =
				separate_benders_cut(instance, y_val, current_scenario; MILP_solver)

			# If feasibility cut
            if !optimality_cut
                new_feasibility_cut = @build_constraint(
                    ν_val + sum(μ_val[e] * y[e] for e in 1:E) <= 0
                )
                MOI.submit(
                    model,
					MOI.LazyConstraint(cb_data),
					new_feasibility_cut
                )

                return nothing
            end

			# Else, optimality cut
			if θ_val[current_scenario] + tol < ν_val + sum(μ_val[e] * y_val[e] for e in 1:E) -
				sum(second_stage_costs[e, current_scenario] * y_val[e] for e in 1:E)
				con = @build_constraint(
					θ[current_scenario] >=
						ν_val + sum(μ_val[e] * y[e] for e in 1:E) - sum(second_stage_costs[e, current_scenario] * y[e] for e in 1:E)
				)
				MOI.submit(model, MOI.LazyConstraint(cb_data), con)
				return nothing
			end
        end
    end

    set_attribute(model, MOI.LazyConstraintCallback(), benders_callback)
    optimize!(model)

	return solution_from_first_stage_forest(value.(y) .> 0.5, instance)
end
