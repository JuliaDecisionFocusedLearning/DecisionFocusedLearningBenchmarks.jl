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
