@testset "Metric interface" begin
    using DecisionFocusedLearningBenchmarks
    using Statistics: mean, std

    @testset "Static generic Metric / MetricStats mechanics" begin
        struct DummyMetricBenchmark <: AbstractStaticBenchmark end
        b = DummyMetricBenchmark()
        dataset = [DataSample(; x=[Float64(i)]) for i in 1:5]

        # per-sample function
        m = Metric(b, "dummy", "x[1] of the sample", (_, sample) -> sample.x[1])
        @test m isa StaticMetric
        @test m isa AbstractStaticMetric
        @test metric_name(m) == "dummy"
        @test metric_benchmark(m) === b
        @test metric_description(m) == "x[1] of the sample"

        stats = evaluate_metric(m, dataset)
        @test stats isa MetricStats
        @test stats.values == Float64[1, 2, 3, 4, 5]
        @test mean_metric(stats) == 3.0
        @test std_metric(stats) == std(1.0:5.0)
        @test quantile_metric(stats, 0.5) == 3.0
    end

    @testset "ObjectiveMetric on ArgmaxBenchmark" begin
        b = ArgmaxBenchmark()
        model = generate_statistical_model(b)
        maximizer = generate_maximizer(b)
        policy(sample) = DataSample(sample; y=maximizer(model(sample.x); sample.context...))

        resolved = generate_dataset(b, 20; seed=0, target_policy=policy)
        om = ObjectiveMetric(b)
        @test om isa StaticMetric
        @test metric_name(om) == "objective"
        @test metric_description(om) ==
            "Objective value of the evaluated policy's decision."
        @test metric_benchmark(om) === b

        stats = evaluate_metric(om, resolved)
        @test length(stats.values) == length(resolved)
        @test stats.values ≈ [objective_value(b, s, s.y) for s in resolved]
        @test mean_metric(stats) ≈ mean(objective_value(b, s, s.y) for s in resolved)
    end

    @testset "Static RelativeGapMetric: mean reproduces compute_gap" begin
        b = ArgmaxBenchmark()
        dataset = generate_dataset(b, 20; seed=0)
        model = generate_statistical_model(b)
        maximizer = generate_maximizer(b)

        gm = RelativeGapMetric(b, model, maximizer)
        @test gm isa StaticMetric
        @test metric_name(gm) == "relative_gap"
        @test metric_description(gm) ==
            "Relative optimality gap of statistical_model+maximizer vs. the sample's target."
        stats = evaluate_metric(gm, dataset)
        @test length(stats.values) == length(dataset)
        @test mean_metric(stats) ≈ compute_gap(b, dataset, model, maximizer)
    end

    @testset "Static RelativeGapMetric on SampleAverageApproximation" begin
        inner = ContextualStochasticArgmaxBenchmark(; n=5, d=3, seed=0)
        saa = SampleAverageApproximation(inner, 20)

        dataset = generate_dataset(saa, 10)
        maximizer = generate_maximizer(saa)
        labeled = map(dataset) do s
            y_saa = maximizer(mean(s.scenarios))
            return DataSample(; s.context..., x=s.x, y=y_saa, extra=s.extra)
        end
        model = generate_statistical_model(saa; seed=0)

        gm = RelativeGapMetric(saa, model, maximizer)
        @test mean_metric(evaluate_metric(gm, labeled)) ≈
            compute_gap(saa, labeled, model, maximizer)
    end

    @testset "Dynamic RewardMetric matches evaluate_policy! returns" begin
        b = DynamicVehicleSchedulingBenchmark()
        env = generate_environment(b; seed=0)
        policy = generate_baseline_policies(b).greedy

        rewards, datasets = evaluate_policy!(policy, env, 4; seed=0)

        rm = RewardMetric(b)  # op=sum by default → episode return
        @test rm isa DynamicMetric
        @test metric_name(rm) == "reward"

        stats = evaluate_metric(rm, datasets)
        @test stats.values ≈ rewards  # per-episode return == evaluate_policy! rewards
        @test mean_metric(stats) ≈ mean(rewards)
        @test std_metric(stats) ≈ std(rewards)

        # per-episode whole-trajectory custom metric (episode length)
        len = Metric(b, "length", "number of steps", (_, ep) -> length(ep))
        @test len isa DynamicMetric
        @test evaluate_metric(len, datasets).values ==
            Float64[length(ep) for ep in datasets]
    end

    @testset "RewardMetric errors when episode samples lack extra.reward" begin
        b = DynamicVehicleSchedulingBenchmark()
        rm = RewardMetric(b)
        episode = [DataSample(; x=[1.0]), DataSample(; x=[2.0])]  # no extra.reward
        @test_throws ErrorException evaluate_metric(rm, [episode])
    end

    @testset "Dynamic RelativeGapMetric: test return vs target return" begin
        b = DynamicVehicleSchedulingBenchmark()
        env = generate_environment(b; seed=0)
        policies = generate_baseline_policies(b)

        seeds = 1:5
        target = [evaluate_policy!(policies.greedy, env; seed=s)[2] for s in seeds]
        test = [evaluate_policy!(policies.lazy, env; seed=s)[2] for s in seeds]

        gm = RelativeGapMetric(b, target)
        @test gm isa DynamicGapMetric
        @test metric_name(gm) == "relative_gap"

        stats = evaluate_metric(gm, test)
        check = is_minimization_problem(b)
        expected = map(target, test) do tgt, tst
            tr = sum(s.extra.reward for s in tgt)
            er = sum(s.extra.reward for s in tst)
            return (check ? er - tr : tr - er) / abs(tr)
        end
        @test stats.values ≈ expected

        # misaligned target/test lengths must error
        @test_throws AssertionError evaluate_metric(gm, test[1:3])
    end

    @testset "Policy comparison via label => collection pairs" begin
        b = DynamicVehicleSchedulingBenchmark()
        env = generate_environment(b; seed=0)
        policies = generate_baseline_policies(b)

        seeds = 1:5
        datasets_greedy = [evaluate_policy!(policies.greedy, env; seed=s)[2] for s in seeds]
        datasets_lazy = [evaluate_policy!(policies.lazy, env; seed=s)[2] for s in seeds]

        rm = RewardMetric(b)
        stats_unlabeled = evaluate_metric(rm, datasets_greedy)
        @test stats_unlabeled.label == ""

        comparison = evaluate_metric(
            rm, policies.greedy.name => datasets_greedy, policies.lazy.name => datasets_lazy
        )
        @test comparison isa Vector{<:MetricStats}
        @test [s.label for s in comparison] == ["Greedy", "Lazy"]
    end
end
