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
            x=randn(rng, 10, 5), θ=rand(rng, 5), y=rand(rng, 10), info="this is an instance"
        )
    end

    sample = random_sample()
    @test sample isa DataSample

    io = IOBuffer()
    show(io, sample)
    @test String(take!(io)) ==
        "DataSample(x=$(sample.x), θ_true=$(sample.θ), y_true=$(sample.y), instance=$(sample.info))"

    # Test StatsBase methods
    using StatsBase:
        ZScoreTransform,
        UnitRangeTransform,
        fit,
        transform,
        transform!,
        reconstruct,
        reconstruct!

    # Create a dataset for testing
    N = 5
    dataset = [random_sample() for _ in 1:N]

    # Test fit with ZScoreTransform
    zt = fit(ZScoreTransform, dataset; dims=2)
    @test zt isa ZScoreTransform

    # Test fit with UnitRangeTransform
    ut = fit(UnitRangeTransform, dataset; dims=2)
    @test ut isa UnitRangeTransform

    # Test transform (non-mutating)
    dataset_zt = transform(zt, dataset)
    @test length(dataset_zt) == length(dataset)
    @test all(d -> d isa DataSample, dataset_zt)

    # Check that other fields are preserved
    for i in 1:N
        @test dataset_zt[i].θ == dataset[i].θ
        @test dataset_zt[i].y == dataset[i].y
        @test dataset_zt[i].info == dataset[i].info
    end

    # Check that features are actually transformed
    @test dataset_zt[1].x != dataset[1].x

    # Test transform! (mutating)
    dataset_copy = deepcopy(dataset)
    original_x = copy(dataset_copy[1].x)
    transform!(ut, dataset_copy)
    @test dataset_copy[1].x != original_x

    # Check that other fields remain unchanged after transform!
    for i in 1:N
        @test dataset_copy[i].θ == dataset[i].θ
        @test dataset_copy[i].y == dataset[i].y
        @test dataset_copy[i].info == dataset[i].info
    end

    # Test reconstruct (non-mutating)
    dataset_reconstructed = reconstruct(zt, dataset_zt)
    @test length(dataset_reconstructed) == length(dataset)

    # Test round-trip consistency (should be close to original)
    for i in 1:N
        @test dataset_reconstructed[i].x ≈ dataset[i].x atol = 1e-10
        @test dataset_reconstructed[i].θ == dataset[i].θ
        @test dataset_reconstructed[i].y == dataset[i].y
        @test dataset_reconstructed[i].info == dataset[i].info
    end

    # Test reconstruct! (mutating)
    reconstruct!(zt, dataset_zt)
    for i in 1:N
        @test dataset_zt[i].x ≈ dataset[i].x atol = 1e-10
    end
end

@testitem "Maximizers" begin
    using DecisionFocusedLearningBenchmarks.Utils: TopKMaximizer
    top_k = TopKMaximizer(3)
    @test top_k([1, 2, 3, 4, 5]) == [0, 0, 1, 1, 1]
    @test top_k([5, 4, 3, 2, 1]) == [1, 1, 1, 0, 0]
    @test_throws(AssertionError, top_k([1, 2]))
end
