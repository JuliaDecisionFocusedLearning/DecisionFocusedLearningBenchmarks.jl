@testitem "FixedSizeShortestPath" begin
    using DecisionFocusedLearningBenchmarks.FixedSizeShortestPath
    using Graphs

    p = 5
    grid_size = (5, 5)
    A = (grid_size[1] - 1) * grid_size[2] + grid_size[1] * (grid_size[2] - 1)
    b = FixedSizeShortestPathBenchmark(; p=p, grid_size=grid_size)

    @test nv(b.graph) == grid_size[1] * grid_size[2]
    @test ne(b.graph) == A

    dataset = generate_dataset(b, 50)
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)

    gap = compute_gap(b, dataset, model, maximizer)
    @test gap >= 0

    for sample in dataset
        x = sample.x
        θ_true = sample.θ
        y_true = sample.y
        @test all(θ_true .< 0)
        @test size(x) == (p,)
        @test length(θ_true) == A
        @test length(y_true) == A
        @test isnothing(sample.info)
        @test all(y_true .== maximizer(θ_true))
        θ = model(x)
        @test length(θ) == length(θ_true)
        y = maximizer(θ)
        @test length(y) == length(y_true)
    end
end
