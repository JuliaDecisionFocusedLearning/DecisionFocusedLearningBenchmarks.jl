# Observation-unit word for plot titles: "episodes" for dynamic benchmarks, "instances" otherwise.
function _metric_unit_word(m::AbstractMetric)
    return metric_benchmark(m) isa AbstractDynamicBenchmark ? "episodes" : "instances"
end

"""
$TYPEDSIGNATURES

Generic default: boxplot of the per-unit metric values in `stats`. Custom metrics can
override this via dispatch on `MetricStats{<:MyMetricType}` for a specialised rendering.
"""
function plot_metric(stats::MetricStats; kwargs...)
    name = metric_name(stats.metric)
    x_label = isempty(stats.label) ? name : stats.label
    n = length(stats.values)
    return StatsPlots.boxplot(
        [x_label],
        stats.values;
        legend=false,
        title="$name across $n $(_metric_unit_word(stats.metric))",
        ylabel=name,
        kwargs...,
    )
end

"""
$TYPEDSIGNATURES

Generic default: boxplot comparing several policies' [`MetricStats`](@ref) for the same
metric, one box per policy (grouped by `label`).
"""
function plot_metric(stats::AbstractVector{<:MetricStats}; kwargs...)
    metric = first(stats).metric
    name = metric_name(metric)
    n = length(first(stats).values)
    labels = reduce(vcat, [fill(s.label, length(s.values)) for s in stats])
    values = reduce(vcat, [s.values for s in stats])
    return StatsPlots.boxplot(
        labels,
        values;
        group=labels,
        legend=false,
        title="$name across $n $(_metric_unit_word(metric))",
        ylabel=name,
        kwargs...,
    )
end
