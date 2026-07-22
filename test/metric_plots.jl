@testset "Metric Plots" begin
    using Plots
    using StatsPlots

    b = ArgmaxBenchmark()
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)
    policy(sample) = DataSample(sample; y=maximizer(model(sample.x); sample.context...))

    datasets = [generate_dataset(b, 20; seed=s, target_policy=policy) for s in 1:3]

    rm = RewardMetric(b)
    stats = evaluate_metric(rm, datasets)

    # Generic default: boxplot of per-dataset values
    fig1 = plot_metric(stats)
    @test fig1 isa Plots.Plot

    # Custom override via dispatch on MetricStats{<:DummyCountMetric}: a user-defined
    # AbstractMetric subtype (not going through the generic Metric wrapper)
    struct DummyCountMetric{B<:AbstractBenchmark} <: AbstractMetric{B}
        bench::B
    end
    DecisionFocusedLearningBenchmarks.metric_benchmark(m::DummyCountMetric) = m.bench
    function DecisionFocusedLearningBenchmarks.evaluate_metric(
        ::DummyCountMetric, dataset::AbstractVector{<:DataSample}
    )
        return length(dataset)
    end
    DecisionFocusedLearningBenchmarks.metric_name(::DummyCountMetric) = "count"

    count_stats = evaluate_metric(DummyCountMetric(b), datasets)

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
    stats_a = evaluate_metric(rm, datasets, "policy A")
    stats_b = evaluate_metric(rm, datasets, "policy B")
    fig3 = plot_metric([stats_a, stats_b])
    @test fig3 isa Plots.Plot
end
