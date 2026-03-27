@testset "Stochastic VSP" begin
    using DecisionFocusedLearningBenchmarks
    using DecisionFocusedLearningBenchmarks.StochasticVehicleScheduling
    using Graphs
    using Plots
    using StableRNGs: StableRNG

    b = StochasticVehicleSchedulingBenchmark(; nb_tasks=25, nb_scenarios=10)

    @test is_exogenous(b)

    N = 2
    K = 3

    # Test unlabeled stochastic dataset: N instances × K scenarios = N*K unlabeled samples
    unlabeled = generate_dataset(b, N; nb_scenarios=K, seed=1)
    @test length(unlabeled) == N * K
    @test hasproperty(unlabeled[1].extra, :scenario)
    @test unlabeled[1].extra.scenario isa VSPScenario

    # Test baseline policies
    policies = generate_baseline_policies(b)
    @test hasproperty(policies, :saa)
    @test hasproperty(policies, :deterministic)
    @test hasproperty(policies, :local_search)

    # Test labeled stochastic dataset with SAA policy
    # N instances, each with K scenarios → N labeled samples
    saa_dataset = generate_dataset(
        b, N; nb_scenarios=K, seed=0, rng=StableRNG(0), target_policy=policies.saa
    )
    @test length(saa_dataset) == N
    @test hasproperty(saa_dataset[1].extra, :scenarios)
    @test saa_dataset[1].extra.scenarios isa Vector{VSPScenario}
    @test length(saa_dataset[1].extra.scenarios) == K
    det_dataset = generate_dataset(
        b, N; nb_scenarios=K, seed=0, rng=StableRNG(0), target_policy=policies.deterministic
    )
    @test length(det_dataset) == N
    @test hasproperty(det_dataset[1].extra, :scenarios)
    @test det_dataset[1].extra.scenarios isa Vector{VSPScenario}
    @test length(det_dataset[1].extra.scenarios) == K
    ls_dataset = generate_dataset(
        b, N; nb_scenarios=K, seed=0, rng=StableRNG(0), target_policy=policies.local_search
    )
    @test length(ls_dataset) == N
    @test hasproperty(ls_dataset[1].extra, :scenarios)
    @test ls_dataset[1].extra.scenarios isa Vector{VSPScenario}
    @test length(ls_dataset[1].extra.scenarios) == K

    # Plots work unchanged
    figure_1 = plot_instance(b, saa_dataset[1])
    @test figure_1 isa Plots.Plot
    figure_2 = plot_solution(b, saa_dataset[1])
    @test figure_2 isa Plots.Plot

    maximizer = generate_maximizer(b)
    model = generate_statistical_model(b)

    # compute_gap runs and returns finite values
    gap = compute_gap(b, saa_dataset, model, maximizer)
    @test isfinite(gap)

    # Features, maximizer output, and feasibility
    for sample in saa_dataset
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

    # Direct solver tests: take an Instance directly (stochastic interface not required)
    direct_sample = generate_dataset(b, 1; seed=42)[1]
    instance = direct_sample.instance

    y_mip = compact_mip(instance)
    @test y_mip isa BitVector

    y_mipl = compact_linearized_mip(instance)
    @test y_mipl isa BitVector

    y_ls = local_search(instance)
    @test y_ls isa BitVector

    y_det = deterministic_mip(instance)
    @test y_det isa BitVector

    anticipative_solver = generate_anticipative_solver(b)
    sample = unlabeled[1]
    y_anticipative = anticipative_solver(sample.scenario; sample.context...)
    @test y_anticipative isa BitVector
end
