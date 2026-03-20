import DecisionFocusedLearningBenchmarks.TwoStageSpanningTree:
    TwoStageSpanningTreeInstance,
    TwoStageSpanningTreeSolution,
    nb_scenarios,
    kruskal,
    solution_from_first_stage_forest
using Graphs: ne, nv, edges, src, dst

has_visualization(::TwoStageSpanningTreeBenchmark) = true

function _plot_grid_graph(
    graph,
    n,
    m,
    weights=nothing;
    edge_colors=fill(:black, ne(graph)),
    edge_widths=fill(1, ne(graph)),
    edge_labels=fill(nothing, ne(graph)),
    δ=0.25,
    δ₂=0.13,
    space_for_legend=0,
)
    node_pos = [((i - 1) % n, floor((i - 1) / n)) for i in 1:nv(graph)]
    function segment(i, j)
        return [node_pos[i][1], node_pos[j][1]], [node_pos[i][2], node_pos[j][2]]
    end
    fig = Plots.plot(;
        axis=([], false),
        ylimits=(-δ, m - 1 + δ + space_for_legend),
        xlimits=(-δ, n - 1 + δ),
        aspect_ratio=:equal,
        leg=:top,
    )
    for (color, width, label, e) in zip(edge_colors, edge_widths, edge_labels, edges(graph))
        Plots.plot!(fig, segment(src(e), dst(e))...; color, width, label)
    end
    Plots.scatter!(fig, node_pos; label=nothing, markersize=15, color=:lightgrey)
    if !isnothing(weights)
        for (w, e) in zip(weights, edges(graph))
            i, j = src(e), dst(e)
            x = (node_pos[j][1] + node_pos[i][1]) / 2
            y = (node_pos[j][2] + node_pos[i][2]) / 2
            j == i + 1 ? (y += δ₂) : (x -= δ₂)
            Plots.annotate!(fig, x, y, Int(w))
        end
    end
    return fig
end

function plot_instance(bench::TwoStageSpanningTreeBenchmark, sample::DataSample; kwargs...)
    (; n, m) = bench
    return _plot_grid_graph(
        sample.instance.graph, n, m, sample.instance.first_stage_costs; kwargs...
    )
end

function plot_solution(bench::TwoStageSpanningTreeBenchmark, sample::DataSample; kwargs...)
    (; n, m) = bench
    (; instance) = sample.context
    y = sample.y
    isnothing(y) && error("sample.y is nothing — provide a labeled sample")

    # Use the evaluation scenario if present, otherwise fall back to first feature scenario
    d_plot = if hasproperty(sample.extra, :scenario)
        sample.extra.scenario
    else
        instance.second_stage_costs[:, 1]
    end

    # Complete first-stage forest to a spanning tree for display
    inst_s = TwoStageSpanningTreeInstance(
        instance.graph, instance.first_stage_costs, reshape(d_plot, :, 1)
    )
    full_sol = solution_from_first_stage_forest(BitVector(y .> 0), inst_s)

    yv, zv = full_sol.y, full_sol.z[:, 1]
    is_labeled_1 = is_labeled_2 = false
    edge_labels = fill(nothing, ne(instance.graph))
    for i in 1:ne(instance.graph)
        if !is_labeled_1 && yv[i]
            edge_labels[i] = "First stage"
            is_labeled_1 = true
        elseif !is_labeled_2 && zv[i]
            edge_labels[i] = "Second stage"
            is_labeled_2 = true
        end
    end
    edge_colors = [
        if yv[i]
            :red
        elseif zv[i]
            :green
        else
            :black
        end for i in 1:ne(instance.graph)
    ]
    edge_widths = [(yv[i] || zv[i]) ? 3 : 1 for i in 1:ne(instance.graph)]
    weights = yv .* instance.first_stage_costs .+ .!yv .* d_plot
    return _plot_grid_graph(
        instance.graph,
        n,
        m,
        weights;
        edge_colors,
        edge_widths,
        edge_labels,
        space_for_legend=0.75,
        kwargs...,
    )
end
