@testitem "Grid graphs" begin
    using InferOptBenchmarks.Utils
    using InferOptBenchmarks.Utils: count_edges, get_path, index_to_coord
    using Graphs

    h = 4
    w = 7
    costs = rand(h, w)

    for acyclic in (true, false)
        g = grid_graph(costs; acyclic=acyclic)
        @test nv(g) == h * w
        @test ne(g) == count_edges(h, w; acyclic)
        @test all(edges(g)) do e
            v1, v2 = src(e), dst(e)
            i1, j1 = index_to_coord(v1, h, w)
            i2, j2 = index_to_coord(v2, h, w)
            a = max(abs(i1 - i2), abs(j1 - j2)) == 1
            b = g.weights[v2, v1] == costs[v2]
            return a && b
        end
        path = get_path(dijkstra_shortest_paths(g, 1).parents, 1, nv(g))
        @test max(h, w) <= length(path) <= h + w
    end
end
