@testitem "Grid graphs" begin
    using DecisionFocusedLearningBenchmarks.Utils
    using DecisionFocusedLearningBenchmarks.Utils: count_edges, get_path, index_to_coord
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

@testitem "DataSample" begin
    using DecisionFocusedLearningBenchmarks
    using StableRNGs

    rng = StableRNG(1234)

    function random_sample()
        return DataSample(;
            x=randn(rng, 10, 5),
            θ_true=rand(rng, 5),
            y_true=rand(rng, 10),
            instance="this is an instance",
        )
    end

    sample = random_sample()
    @test sample isa DataSample

    io = IOBuffer()
    show(io, sample)
    @test String(take!(io)) ==
        "DataSample(x=$(sample.x), θ_true=$(sample.θ_true), y_true=$(sample.y_true), instance=$(sample.instance))"
end

@testitem "Maximizers" begin
    using DecisionFocusedLearningBenchmarks.Utils: TopKMaximizer
    top_k = TopKMaximizer(3)
    @test top_k([1, 2, 3, 4, 5]) == [0, 0, 1, 1, 1]
    @test top_k([5, 4, 3, 2, 1]) == [1, 1, 1, 0, 0]
    @test_throws(AssertionError, top_k([1, 2]))
end
