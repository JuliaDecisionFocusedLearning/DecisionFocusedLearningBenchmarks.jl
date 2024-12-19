@testitem "Portfolio Optimization" begin
    using DecisionFocusedLearningBenchmarks

    d = 50
    p = 5
    b = PortfolioOptimizationBenchmark(; d=d, p=p)

    dataset = generate_dataset(b, 50)
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)

    for sample in dataset
        (; x, θ_true, y_true) = sample
        @test size(x) == (p,)
        @test length(θ_true) == d
        @test length(y_true) == d
        @test isnothing(sample.instance)
        @test all(y_true .== maximizer(θ_true))

        θ = model(x)
        @test length(θ) == d

        y = maximizer(θ)
        @test length(y) == d
        @test sum(y) <= 1 + 1e-6
    end
end
