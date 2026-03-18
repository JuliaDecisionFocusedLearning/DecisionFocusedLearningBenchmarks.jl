"""
$TYPEDSIGNATURES

Solve the separation problem using the MILP formulation.
"""
function MILP_separation_problem(graph, weights; MILP_solver, tol=1e-6)
	V = nv(graph)
	E = ne(graph)

	model = Model(MILP_solver)
	set_silent(model)

	@variable(model, α[v in 1:V], Bin)
	@variable(model, β[e in 1:E], Bin)

	@objective(
		model, Min,
		sum(α[v] for v in 1:V) - 1 - sum(weights[e] * β[e] for e in 1:E)
	)

	@constraint(model, [(e, edge) in enumerate(edges(graph))], 2 * β[e] <= α[dst(edge)] + α[src(edge)])

	@constraint(model, sum(α[v] for v in 1:V) >= 1)
	@constraint(model, sum(α[v] for v in 1:V) <= V - 1)
	
	optimize!(model)
	found = objective_value(model) + tol <= 0
	return found, value.(β) .> 0.5, sum(value.(α))
end

function build_flow_graph(graph, weights; infinity=1e6)
	V = nv(graph)
	E = ne(graph)
	
	# A = 3 * E + V
	VV = 2 + E + V

	o = 1
	d = 2

	sources = vcat(
		fill(o, E),
		[2 + e for e in 1:E],
		[2 + e for e in 1:E],
		[2 + E + v for v in 1:V]
	)
	destinations = vcat(
		[2 + e for e in 1:E],
		[2 + E + src(e) for e in edges(graph)],
		[2 + E + dst(e) for e in edges(graph)],
		fill(d, V)
	)
	costs = vcat(
		[weights[e] for e in 1:E],
		fill(infinity, 2 * E),
		ones(V)
	)

	return sources, destinations, costs
end

"""
$TYPEDSIGNATURES

Solve the separation problem using the min cut formulation.
"""
function cut_separation_problem(graph, weights; MILP_solver=GLPK.Optimizer, tol=1e-6)
	sources, destinations, costs = build_flow_graph(graph, weights)

	A = 3 * ne(graph) + nv(graph)
	V = 2 + ne(graph) + nv(graph)
	@assert A == length(costs)

	vertex_range = (2 + ne(graph) + 1):V
	edge_range = 1:ne(graph)

	model = Model(MILP_solver)

	@variable(model, β[a in 1:A], Bin)
	@variable(model, α[v in 1:V], Bin)

	@objective(
		model, Min,
		sum(costs[a] * β[a] for a in 1:A)
	)

	@constraint(model, α[1] == 1)
	@constraint(model, α[2] == 0)

	@constraint(
		model,
		[a in 1:A],
		β[a] >= α[sources[a]] - α[destinations[a]]
	)

	@constraint(model, sum(α[v] for v in vertex_range) >= 1)

	optimize!(model)

	min_cut_value = objective_value(model)

    found = min_cut_value + tol < nv(graph)

    return found, value.(β)[edge_range] .< 0.5, sum(value.(α)[vertex_range])
end

"""
$TYPEDSIGNATURES

Returns the optimal solution using a cut generation algorithm with custom separation problem solver.
"""
function cut_generation(
    instance::TwoStageSpanningTreeInstance;
	separation_problem=MILP_separation_pb,
    MILP_solver=GLPK.Optimizer,
	verbose=true
)
	# Unpack fields
	(; graph, first_stage_costs, second_stage_costs) = instance
	S = nb_scenarios(instance)
	E = ne(graph)
	V = nv(graph)

	# Initialize model and link to solver
	model = Model(MILP_solver)

	# Add variables
    @variable(model, y[e in 1:E], Bin)
    @variable(model, z[e in 1:E, s in 1:S], Bin)

	# Add an objective function
	@expression(
		model,
		first_stage_objective,
		sum(first_stage_costs[e] * y[e] for e in 1:E)
	)
	@expression(
		model,
		second_stage_objective,
		sum(second_stage_costs[e, s] * z[e, s] for e in 1:E for s in 1:S) / S
	)
    @objective(model, Min, first_stage_objective + second_stage_objective)

	# Add constraints
	@constraint(
		model, [s in 1:S],
		sum(y[e] + z[e, s] for e in 1:E) == V - 1
	)

    call_back_counter = 0

    function my_callback_function(cb_data)
        call_back_counter += 1

        y_val = callback_value.(cb_data, y)
        z_val = callback_value.(cb_data, z)

        for current_s in 1:S
			weights = [y_val[e] + z_val[e, current_s] for e in 1:E]

            found, Y, YY = separation_problem(
				graph, weights; MILP_solver=MILP_solver
			)

            if found
                new_constraint = @build_constraint(
                    sum(y[e] + z[e, current_s] for e in 1:E if Y[e]) <= YY - 1
                )
                MOI.submit(
                    model, MOI.LazyConstraint(cb_data), new_constraint
                )
                return
            end
        end
    end

	set_attribute(model, MOI.LazyConstraintCallback(), my_callback_function)

	optimize!(model)
	verbose && @info "Optimal solution found after $call_back_counter cuts"
    return TwoStageSpanningTreeSolution(value.(y) .> 0.5, value.(z) .> 0.5)
end
