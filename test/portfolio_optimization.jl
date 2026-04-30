@testset "Portfolio Optimization" begin
    using DecisionFocusedLearningBenchmarks

    d = 50
    p = 5
    b = PortfolioOptimizationBenchmark(; d=d, p=p)

    dataset = generate_dataset(b, 50)
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)

    for sample in dataset
        x = sample.x
        θ_true = sample.θ
        y_true = sample.y
        @test size(x) == (p,)
        @test length(θ_true) == d
        @test length(y_true) == d
        @test isempty(sample.context)
        @test all(y_true .== maximizer(θ_true))

        θ = model(x)
        @test length(θ) == d

        y = maximizer(θ)
        @test length(y) == d
        @test sum(y) <= 1 + 1e-6
    end

    gap = compute_gap(b, dataset[1:5], model, maximizer)
    @test isfinite(gap)

    @testset "Plots" begin
        using Plots
        @test has_visualization(b)
        fig1 = plot_instance(b, dataset[1])
        @test fig1 isa Plots.Plot
        fig2 = plot_solution(b, dataset[1])
        @test fig2 isa Plots.Plot
        fig3 = plot_solution(b, dataset[1], dataset[2].y)
        @test fig3 isa Plots.Plot
    end
end
