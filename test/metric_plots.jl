@testset "Metric Plots" begin
    using Plots
    using StatsPlots

    b = ArgmaxBenchmark()
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)
    policy(sample) = DataSample(sample; y=maximizer(model(sample.x); sample.context...))

    dataset = generate_dataset(b, 20; seed=0, target_policy=policy)
    stats = evaluate_metric(ObjectiveMetric(b), dataset)

    # Generic default: boxplot of per-sample values
    fig1 = plot_metric(stats)
    @test fig1 isa Plots.Plot

    # Custom override via dispatch on MetricStats{<:MyMetric}: a user-defined static metric
    struct DummyCountMetric{B<:AbstractStaticBenchmark} <: AbstractStaticMetric{B}
        bench::B
    end
    DecisionFocusedLearningBenchmarks.metric_benchmark(m::DummyCountMetric) = m.bench
    DecisionFocusedLearningBenchmarks.metric_name(::DummyCountMetric) = "count"
    (::DummyCountMetric)(::DataSample) = 1.0

    count_stats = evaluate_metric(DummyCountMetric(b), dataset)
    @test count_stats.values == fill(1.0, length(dataset))

    custom_plot_called = Ref(false)
    function DecisionFocusedLearningBenchmarks.plot_metric(
        s::MetricStats{<:DummyCountMetric}; kwargs...
    )
        custom_plot_called[] = true
        return StatsPlots.boxplot(["custom count"], s.values; kwargs...)
    end

    fig2 = plot_metric(count_stats)
    @test fig2 isa Plots.Plot
    @test custom_plot_called[]

    # Comparing several policies: Vector{<:MetricStats}, grouped boxplot by label
    dataset_b = generate_dataset(b, 20; seed=1, target_policy=policy)
    stats_a = evaluate_metric(ObjectiveMetric(b), dataset, "policy A")
    stats_b = evaluate_metric(ObjectiveMetric(b), dataset_b, "policy B")
    fig3 = plot_metric([stats_a, stats_b])
    @test fig3 isa Plots.Plot
end
