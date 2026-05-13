@testset "Contextual Stochastic VSP" begin
    using DecisionFocusedLearningBenchmarks
    using DecisionFocusedLearningBenchmarks.StochasticVehicleScheduling
    using Graphs
    using Plots
    using StableRNGs: StableRNG

    b = ContextualStochasticVehicleSchedulingBenchmark(; nb_tasks=15, nb_scenarios=4)

    @test is_exogenous(b)

    N = 2
    M = 3
    K = 2

    # N instances × M contexts × K scenarios = N*M*K unlabeled samples
    unlabeled = generate_dataset(b, N; contexts_per_instance=M, nb_scenarios=K, seed=1)
    @test length(unlabeled) == N * M * K
    @test hasproperty(unlabeled[1].extra, :scenario)
    @test unlabeled[1].extra.scenario isa VSPScenario

    # Each sample carries μ/σ in context
    @test hasproperty(unlabeled[1].context, :district_μ)
    @test hasproperty(unlabeled[1].context, :district_σ)
    @test length(unlabeled[1].district_μ) == length(unlabeled[1].instance.city.districts)
    @test length(unlabeled[1].district_σ) == length(unlabeled[1].instance.city.districts)

    # Samples sharing an instance have identical city, but different μ/σ across contexts
    same_instance_block = unlabeled[1:(M * K)]
    cities = unique(s.instance.city for s in same_instance_block)
    @test length(cities) == 1
    μs = [s.district_μ for s in same_instance_block[1:K:end]]  # one per context
    @test length(unique(μs)) == M

    # Features are 5 × ne(graph), Float32
    sample = unlabeled[1]
    E = ne(sample.instance.graph)
    @test size(sample.x) == (5, E)
    @test eltype(sample.x) === Float32

    # Statistical model + maximizer pipeline
    model = generate_statistical_model(b; seed=0)
    maximizer = generate_maximizer(b)
    θ = model(sample.x)
    @test length(θ) == E
    y = maximizer(θ; sample.context...)
    @test y isa BitVector
    @test length(y) == E

    # Baseline policies
    policies = generate_baseline_policies(b)
    @test hasproperty(policies, :saa)
    @test hasproperty(policies, :deterministic)
    @test hasproperty(policies, :local_search)

    # Labeled dataset via SAA policy: N*M labeled samples
    saa_dataset = generate_dataset(
        b,
        N;
        contexts_per_instance=M,
        nb_scenarios=K,
        seed=0,
        rng=StableRNG(0),
        target_policy=policies.saa,
    )
    @test length(saa_dataset) == N * M
    @test hasproperty(saa_dataset[1].extra, :scenarios)
    @test length(saa_dataset[1].extra.scenarios) == K
    @test saa_dataset[1].y isa BitVector

    # Plot extension
    @test has_visualization(b)
    figure_1 = plot_context(b, saa_dataset[1])
    @test figure_1 isa Plots.Plot
    figure_2 = plot_sample(b, saa_dataset[1])
    @test figure_2 isa Plots.Plot

    # compute_gap returns a finite number
    gap = compute_gap(b, saa_dataset, model, maximizer)
    @test isfinite(gap)

    # Objective value: ensure regenerating scenario from same context reproduces consistent value
    rng = StableRNG(7)
    fresh_scenario = generate_scenario(b, rng; sample.context...)
    @test fresh_scenario isa VSPScenario
    obj = objective_value(b, sample, y, fresh_scenario)
    @test isfinite(obj)

    # Anticipative solvers (1-scenario)
    anticipative_solver = generate_anticipative_solver(b)
    y_anti = anticipative_solver(sample.scenario; sample.context...)
    @test y_anti isa BitVector

    parametric_solver = generate_parametric_anticipative_solver(b)
    θ_zero = zeros(E)
    y_zero = parametric_solver(θ_zero, sample.scenario; sample.context...)
    @test y_zero == y_anti
end
