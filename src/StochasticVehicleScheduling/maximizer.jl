"""
$TYPEDSIGNATURES

Given arcs weights θ, solve the deterministic VSP problem associated to `instance`.
"""
function vsp_maximizer(
    θ::AbstractVector; instance::Instance, model_builder=highs_model, silent=true
)
    (; graph) = instance

    model = model_builder()
    silent && set_silent(model)

    nb_nodes = nv(graph)
    job_indices = 2:(nb_nodes - 1)

    @variable(model, y[i=1:nb_nodes, j=1:nb_nodes; has_edge(graph, i, j)], Bin)

    @objective(
        model,
        Max,
        sum(θ[i] * y[src(edge), dst(edge)] for (i, edge) in enumerate(edges(graph)))
    )

    @constraint(
        model,
        flow[i in job_indices],
        sum(y[j, i] for j in inneighbors(graph, i)) ==
            sum(y[i, j] for j in outneighbors(graph, i))
    )
    @constraint(
        model, demand[i in job_indices], sum(y[j, i] for j in inneighbors(graph, i)) == 1
    )

    optimize!(model)

    solution = falses(ne(graph))
    for (i, edge) in enumerate(edges(graph))
        if isapprox(value(y[edge.src, edge.dst]), 1; atol=1e-3)
            solution[i] = true
        end
    end

    return solution
end
