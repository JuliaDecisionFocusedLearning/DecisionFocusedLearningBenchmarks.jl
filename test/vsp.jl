@testset "Stochastic VSP" begin
    using DecisionFocusedLearningBenchmarks
    using DecisionFocusedLearningBenchmarks.StochasticVehicleScheduling
    using Graphs
    using Plots
    using StableRNGs: StableRNG

    b = StochasticVehicleSchedulingBenchmark(; nb_tasks=25, nb_scenarios=10)

    N = 5

    # Helper to build a target_policy that wraps a given algorithm
    function make_svs_target_policy(algorithm)
        return sample ->
            DataSample(; sample.context..., x=sample.x, y=algorithm(sample.instance))
    end

    col_gen_policy = make_svs_target_policy(column_generation_algorithm)
    mip_policy = make_svs_target_policy(compact_mip)
    mipl_policy = make_svs_target_policy(compact_linearized_mip)
    local_search_policy = make_svs_target_policy(local_search)
    deterministic_policy = make_svs_target_policy(deterministic_mip)

    dataset = generate_dataset(b, N; seed=0, rng=StableRNG(0), target_policy=col_gen_policy)
    mip_dataset = generate_dataset(b, N; seed=0, rng=StableRNG(0), target_policy=mip_policy)
    mipl_dataset = generate_dataset(
        b, N; seed=0, rng=StableRNG(0), target_policy=mipl_policy
    )
    local_search_dataset = generate_dataset(
        b, N; seed=0, rng=StableRNG(0), target_policy=local_search_policy
    )
    deterministic_dataset = generate_dataset(
        b, N; seed=0, rng=StableRNG(0), target_policy=deterministic_policy
    )
    @test length(dataset) == N

    figure_1 = plot_instance(b, dataset[1])
    @test figure_1 isa Plots.Plot
    figure_2 = plot_solution(b, dataset[1])
    @test figure_2 isa Plots.Plot

    maximizer = generate_maximizer(b)
    model = generate_statistical_model(b)

    gap = compute_gap(b, dataset, model, maximizer)
    gap_mip = compute_gap(b, mip_dataset, model, maximizer)
    gap_mipl = compute_gap(b, mipl_dataset, model, maximizer)
    gap_local_search = compute_gap(b, local_search_dataset, model, maximizer)
    gap_deterministic = compute_gap(b, deterministic_dataset, model, maximizer)

    @test gap_mip ≈ gap_mipl rtol = 1e-2
    @test gap_mip >= gap_local_search
    @test gap_mip >= gap
    @test gap_local_search >= gap_deterministic

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
