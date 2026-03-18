"""
	plot_grid_graph(graph, n, m, weights=nothing)

# Arguments
- `graph`: grid graph to plot
- `n`: n dimension
- `m`: m dimension
- `weights`: edge weights to display (optional)
"""
function plot_grid_graph(
	graph, n, m, weights=nothing;
	show_node_indices=false, δ=0.25, δ₂=0.13,
	edge_colors=fill(:black, ne(graph)),
	edge_widths=fill(1, ne(graph)),
	edge_labels=fill(nothing, ne(graph)),
	space_for_legend=0
)
	l = [((i - 1) % n, floor((i - 1) / n)) for i in 1:nv(graph)]
	function args_from_ij(i, j)
		return [l[i][1], l[j][1]], [l[i][2], l[j][2]]
	end
	f = Plots.plot(; axis=([], false), ylimits=(-δ, m-1+δ+space_for_legend), xlimits=(-δ, n-1+δ), aspect_ratio=:equal, leg=:top)
	for (color, width, label, e) in zip(edge_colors, edge_widths, edge_labels, edges(graph))
		Plots.plot!(f, args_from_ij(src(e), dst(e)); color, width, label)
	end
	series_annotations = show_node_indices ? (1:nv(g)) : nothing
	Plots.scatter!(f, l; series_annotations, label=nothing, markersize=15, color=:lightgrey)
	if !isnothing(weights)
		for (w, e) in zip(weights, edges(graph))
			i, j = src(e), dst(e)
			x, y = (l[j] .+ l[i]) ./ 2
			if j == i + 1
				y += δ₂
			else
				x -= δ₂
			end
			Plots.annotate!(f, x, y, Int(w))
		end
	end
	return f
end

"""
$TYPEDSIGNATURES

Plot the two-stage tree from `solution` for requested `scenario`.
"""
function plot_scenario(
	solution::TwoStageSpanningTreeSolution, instance::TwoStageSpanningTreeInstance, scenario;
	show_node_indices=false, δ=0.25, δ₂=0.16, n, m
)
	(; graph, first_stage_costs, second_stage_costs) = instance
	first_stage_forest = solution.y
	second_stage_forests = solution.z

	is_labeled_1 = false
	is_labeled_2 = false
	edge_labels = fill("", ne(graph))

	S = nb_scenarios(instance)

	for e in 1:ne(graph)
		b1 = first_stage_forest[e]
		b2 = second_stage_forests[e, scenario]
		if !is_labeled_1 && b1
			edge_labels[e] = "First stage forest"
			is_labeled_1 = true
		elseif !is_labeled_2 && b2
			edge_labels[e] = "Second stage forest (scenario $scenario/$S)"
			is_labeled_2 = true
		end
	end

	edge_colors = [e1 ? :red : e2 ? :green : :black for (e1, e2) in zip(first_stage_forest, second_stage_forests[:, scenario])]
	edge_widths = [e1 || e2 ? 3 : 1 for (e1, e2) in zip(first_stage_forest, second_stage_forests[:, scenario])]
	weights = first_stage_forest .* first_stage_costs + .!first_stage_forest .* second_stage_costs[:, scenario]
	return plot_grid_graph(
		graph, n, m, weights; show_node_indices, δ, δ₂, edge_colors, edge_widths, edge_labels, space_for_legend=3δ
	)
end
