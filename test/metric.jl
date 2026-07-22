@testset "Metric interface" begin
    using DecisionFocusedLearningBenchmarks
    using Statistics: mean, std

    @testset "AbstractMetric / Metric / MetricStats mechanics" begin
        struct DummyMetricBenchmark <: AbstractStaticBenchmark end
        b = DummyMetricBenchmark()
        dataset = [DataSample(; x=[Float64(i)]) for i in 1:5]

        m = Metric(b, "dummy", "sum of x[1]", (_, ds) -> sum(s.x[1] for s in ds))
        @test metric_name(m) == "dummy"
        @test metric_benchmark(m) === b
        @test evaluate_metric(m, dataset) == sum(1:5)

        stats = evaluate_metric(m, [dataset, dataset, reverse(dataset)])
        @test stats isa MetricStats
        @test stats.values == [15.0, 15.0, 15.0]
        @test mean_metric(stats) == 15.0
        @test std_metric(stats) == 0.0
        @test quantile_metric(stats, 0.5) == 15.0
    end

    @testset "RewardMetric on ArgmaxBenchmark" begin
        b = ArgmaxBenchmark()
        model = generate_statistical_model(b)
        maximizer = generate_maximizer(b)
        policy(sample) = DataSample(sample; y=maximizer(model(sample.x); sample.context...))

        resolved = generate_dataset(b, 20; seed=0, target_policy=policy)
        expected = mean(objective_value(b, s, s.y) for s in resolved)

        rm = RewardMetric(b)
        @test metric_benchmark(rm) === b
        @test evaluate_metric(rm, resolved) ≈ expected
    end

    @testset "RelativeGapMetric reproduces compute_gap" begin
        b = ArgmaxBenchmark()
        dataset = generate_dataset(b, 20; seed=0)
        model = generate_statistical_model(b)
        maximizer = generate_maximizer(b)

        gap_legacy = compute_gap(b, dataset, model, maximizer)
        gm = RelativeGapMetric(b, model, maximizer)
        @test evaluate_metric(gm, dataset) == gap_legacy
    end

    @testset "RelativeGapMetric reproduces compute_gap (SampleAverageApproximation)" begin
        inner = ContextualStochasticArgmaxBenchmark(; n=5, d=3, seed=0)
        saa = SampleAverageApproximation(inner, 20)

        dataset = generate_dataset(saa, 10)
        maximizer = generate_maximizer(saa)
        labeled = map(dataset) do s
            y_saa = maximizer(mean(s.scenarios))
            return DataSample(; s.context..., x=s.x, y=y_saa, extra=s.extra)
        end
        model = generate_statistical_model(saa; seed=0)

        gap_legacy = compute_gap(saa, labeled, model, maximizer)
        gm = RelativeGapMetric(saa, model, maximizer)
        @test evaluate_metric(gm, labeled) == gap_legacy
    end

    @testset "RewardMetric on dynamic benchmark matches evaluate_policy!" begin
        b = DynamicVehicleSchedulingBenchmark()
        env = generate_environment(b; seed=0)
        policy = generate_baseline_policies(b).greedy

        rewards, datasets = evaluate_policy!(policy, env, 4; seed=0)

        rm = RewardMetric(b; op=sum)
        vals = [evaluate_metric(rm, d) for d in datasets]
        @test vals ≈ rewards

        stats = evaluate_metric(rm, datasets)
        @test stats.values ≈ rewards
        @test mean_metric(stats) ≈ mean(rewards)
        @test std_metric(stats) ≈ std(rewards)
    end

    @testset "MetricStats label / policy comparison" begin
        b = DynamicVehicleSchedulingBenchmark()
        env = generate_environment(b; seed=0)
        policies = generate_baseline_policies(b)

        seeds = 1:5
        datasets_greedy = [evaluate_policy!(policies.greedy, env; seed=s)[2] for s in seeds]
        datasets_lazy = [evaluate_policy!(policies.lazy, env; seed=s)[2] for s in seeds]

        rm = RewardMetric(b; op=sum)
        stats_unlabeled = evaluate_metric(rm, datasets_greedy)
        @test stats_unlabeled.label == ""

        stats_greedy = evaluate_metric(rm, datasets_greedy, policies.greedy.name)
        stats_lazy = evaluate_metric(rm, datasets_lazy, policies.lazy.name)
        @test stats_greedy.label == "Greedy"
        @test stats_lazy.label == "Lazy"

        comparison = [stats_greedy, stats_lazy]
        @test comparison isa Vector{<:MetricStats}

        # one-call convenience: pairs of label => datasets
        comparison2 = evaluate_metric(
            rm, policies.greedy.name => datasets_greedy, policies.lazy.name => datasets_lazy
        )
        @test comparison2 isa Vector{<:MetricStats}
        @test [s.label for s in comparison2] == ["Greedy", "Lazy"]
        @test [s.values for s in comparison2] == [stats_greedy.values, stats_lazy.values]
    end
end
