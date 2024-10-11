@testitem "Portfolio Optimization" begin
    using DecisionFocusedLearningBenchmarks

    b = PortfolioOptimizationBenchmark()

    dataset = generate_dataset(b, 100)
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)
end
