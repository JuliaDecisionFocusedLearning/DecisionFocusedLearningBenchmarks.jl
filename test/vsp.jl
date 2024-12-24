@testitem "Stochastic VSP" begin
    using DecisionFocusedLearningBenchmarks
    using DecisionFocusedLearningBenchmarks.StochasticVehicleScheduling
    using Graphs

    b = StochasticVehicleSchedulingBenchmark(; nb_tasks=25, nb_scenarios=10)

    N = 10
    dataset = generate_dataset(b, N; compute_solutions=true, seed=0)
    mip_dataset = generate_dataset(
        b, N; compute_solutions=true, seed=0, algorithm=compact_mip
    )
    mipl_dataset = generate_dataset(
        b, N; compute_solutions=true, seed=0, algorithm=compact_linearized_mip
    )
    @test length(dataset) == N

    figure_1 = plot_instance(b, dataset[1])
    @test figure_1 isa Plots.Plot
    figure_2 = plot_solution(b, dataset[1])
    @test figure_2 isa Plots.Plot

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
