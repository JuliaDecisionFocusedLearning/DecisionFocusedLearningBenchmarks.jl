@testset "ContextualStochasticArgmaxBenchmark" begin
    using DecisionFocusedLearningBenchmarks

    b = ContextualStochasticArgmaxBenchmark(; n=5, d=3, seed=0)

    # Unlabeled: N instances × M contexts × K scenarios = N*M*K samples
    dataset = generate_dataset(b, 10; contexts_per_instance=2, nb_scenarios=4)
    @test length(dataset) == 80
    sample = first(dataset)
    @test size(sample.x) == (8,)                                     # n+d
    @test sample.x ≈ vcat(sample.c_base, sample.x_raw)              # features = [c_base; x_raw]
    @test sample.y === nothing
    @test sample.scenario isa AbstractVector{Float32} && length(sample.scenario) == 5

    # Maximizer and model
    maximizer = generate_maximizer(b)
    model = generate_statistical_model(b; seed=0)
    @test sum(maximizer(sample.scenario)) ≈ 1.0  # one-hot

    # Test with anticipative target_policy
    policy =
        (ctx_sample, scenarios) -> [
            DataSample(;
                ctx_sample.context...,
                x=ctx_sample.x,
                y=maximizer(s),
                extra=(; ctx_sample.extra..., scenario=s),
            ) for s in scenarios
        ]
    labeled = generate_dataset(b, 5; nb_scenarios=3, target_policy=policy)
    @test length(labeled) == 15
    @test sum(first(labeled).y) ≈ 1.0    # one-hot label

    # Reproducibility
    d1 = generate_dataset(b, 5; nb_scenarios=2, seed=42)
    b2 = ContextualStochasticArgmaxBenchmark(; n=5, d=3, seed=0)
    d2 = generate_dataset(b2, 5; nb_scenarios=2, seed=42)
    @test first(d1).x ≈ first(d2).x
end

@testset "Parametric Anticipative Solver - ContextualStochasticArgmax" begin
    using DecisionFocusedLearningBenchmarks

    b = ContextualStochasticArgmaxBenchmark(; n=5, d=3, seed=0)
    dataset = generate_dataset(b, 2; contexts_per_instance=1, nb_scenarios=1)
    sample = first(dataset)
    scenario = generate_scenario(b, StableRNG(0); sample.context...)

    solver = generate_anticipative_solver(b)
    parametric_solver = generate_parametric_anticipative_solver(b)

    # 1. Zero perturbation equivalence
    θ_zero = zeros(eltype(scenario), size(scenario))
    @test parametric_solver(θ_zero, scenario; sample.context...) ==
        solver(scenario; sample.context...)

    # 2. Extreme perturbation
    θ_extreme = zeros(eltype(scenario), size(scenario))
    θ_extreme[1] = 1000.0  # Force dimension 1
    y_extreme = parametric_solver(θ_extreme, scenario; sample.context...)

    @test y_extreme[1] == 1.0     # Only dimension 1 should be active
    @test sum(y_extreme) ≈ 1.0    # One-hot preserved
end

@testset "csa_saa_policy" begin
    using DecisionFocusedLearningBenchmarks

    b = ContextualStochasticArgmaxBenchmark(; n=5, d=3, seed=0)
    policies = generate_baseline_policies(b)

    labeled = generate_dataset(b, 3; nb_scenarios=4, target_policy=policies.saa)
    @test length(labeled) == 3                               # one sample per context (SAA aggregates)
    @test sum(first(labeled).y) ≈ 1.0                       # one-hot label
    @test length(first(labeled).extra.scenarios) == 4       # scenarios stored in extra
end

@testset "SampleAverageApproximation wrapper on ContextualStochasticArgmax" begin
    using DecisionFocusedLearningBenchmarks
    using Statistics: mean

    inner = ContextualStochasticArgmaxBenchmark(; n=5, d=3, seed=0)
    saa = SampleAverageApproximation(inner, 20)

    # Static instances: each sample has x and stored scenarios (no θ)
    dataset = generate_dataset(saa, 10)
    @test length(dataset) == 10
    sample = first(dataset)
    @test size(sample.x) == (8,)
    @test sample.θ === nothing
    @test length(sample.extra.scenarios) == 20    # nb_scenarios

    # Label by SAA-optimal: y = argmax(mean(scenarios)) for linear objectives
    maximizer = generate_maximizer(saa)
    labeled = map(dataset) do s
        y_saa = maximizer(mean(s.scenarios))
        DataSample(; s.context..., x=s.x, y=y_saa, extra=s.extra)
    end
    @test sum(first(labeled).y) ≈ 1.0

    # compute_gap averages over stored scenarios via objective_value override
    model = generate_statistical_model(saa; seed=0)
    gap = compute_gap(saa, labeled, model, maximizer)
    @test isfinite(gap)
end

@testset "csa_objective_value_error" begin
    using DecisionFocusedLearningBenchmarks

    b = ContextualStochasticArgmaxBenchmark(; n=5, d=3, seed=0)
    maximizer = generate_maximizer(b)
    # Sample with neither :scenario nor :scenarios in extra → objective_value should error
    s = DataSample(; x=randn(Float32, 8), y=maximizer(randn(Float32, 5)))
    @test_throws Exception objective_value(b, s, s.y)
end
