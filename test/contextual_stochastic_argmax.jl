@testset "ContextualStochasticArgmaxBenchmark" begin
    using DecisionFocusedLearningBenchmarks

    b = ContextualStochasticArgmaxBenchmark(; n=5, d=3, seed=0)

    # Unlabeled: N instances × M contexts × K scenarios = N*M*K samples
    dataset = generate_dataset(b, 10; nb_contexts=2, nb_scenarios=4)
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
