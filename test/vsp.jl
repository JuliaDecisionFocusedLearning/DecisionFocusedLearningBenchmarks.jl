@testitem "Stochastic VSP" begin
    using DecisionFocusedLearningBenchmarks
    using DecisionFocusedLearningBenchmarks.StochasticVehicleScheduling
    using Graphs

    b = StochasticVehicleSchedulingBenchmark(; nb_tasks=25, nb_scenarios=10)

    N = 50
    dataset = generate_dataset(b, N)
    @test length(dataset) == N

    maximizer = generate_maximizer(b)
    model = generate_statistical_model(b)

    for sample in dataset
        x = sample.x
        instance = sample.instance
        E = ne(instance.graph)
        @test size(x) == (20, E)
        θ = model(x)
        @test length(θ) == E
        y = maximizer(θ; instance=instance)
        @test length(y) == E
        solution = StochasticVehicleScheduling.Solution(y, instance)
        @test is_feasible(solution, instance)
    end
end
