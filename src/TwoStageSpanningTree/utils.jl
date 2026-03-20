"""
$TYPEDSIGNATURES

Kruskal's algorithm.
Same as [`Graphs.kruskal_mst`](https://juliagraphs.org/Graphs.jl/dev/algorithms/spanningtrees/#Graphs.kruskal_mst), but also returns the value of the tree,
and a binary vector instead of a vecror of edges.
"""
function kruskal(g::AbstractGraph, weights::AbstractVector; minimize::Bool=true)
    connected_vs = IntDisjointSets(nv(g))

    tree = falses(ne(g))

    edge_list = collect(edges(g))
    order = sortperm(weights; rev=!minimize)
    value = 0.0

    tree_size = 0

    for (e_ind, e) in zip(order, edge_list[order])
        if !in_same_set(connected_vs, src(e), dst(e))
            union!(connected_vs, src(e), dst(e))
            tree[e_ind] = true
            tree_size += 1
            value += weights[e_ind]
            (tree_size >= nv(g) - 1) && break
        end
    end

    return (; value, tree)
end

function is_spanning_tree(tree_candidate::BitVector, graph::AbstractGraph)
    edge_list = [e for (i, e) in enumerate(edges(graph)) if tree_candidate[i]]
    subgraph = induced_subgraph(graph, edge_list)[1]
    return !is_cyclic(subgraph) && nv(subgraph) == nv(graph)
end

"""
$TYPEDSIGNATURES

Return the index of edge `e` in `collect(edges(graph))`.
"""
function edge_index(graph::AbstractGraph, e::AbstractEdge)
    for (i, f) in enumerate(edges(graph))
        (src(f) == src(e) && dst(f) == dst(e)) && return i
    end
    return error("Edge $(e) not found in graph")
end

"""
$TYPEDSIGNATURES

Return the maximum weight forest of `g` with respect to edge weights `θ`.
Edges with non-positive weight are excluded.
"""
function maximum_weight_forest(g::AbstractGraph, θ::AbstractVector)
    edge_list = collect(edges(g))
    order = sortperm(θ; rev=true)
    forest = falses(ne(g))
    connected_vs = IntDisjointSets(nv(g))
    for i in order
        θ[i] <= 0 && break
        e = edge_list[i]
        if !in_same_set(connected_vs, src(e), dst(e))
            union!(connected_vs, src(e), dst(e))
            forest[i] = true
        end
    end
    return forest
end
