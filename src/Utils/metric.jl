"""
$TYPEDEF

Abstract root type for evaluation metrics, used to evaluate and compare [`Policy`](@ref)s
on a benchmark. Parametrized by the benchmark type `B` the metric is bound to — a metric is
always tied to one specific benchmark instance (see [`metric_benchmark`](@ref)), since a
dataset only ever makes sense with respect to the benchmark that produced it.

Concrete subtypes implement

    evaluate_metric(metric, dataset::AbstractVector{<:DataSample}) -> Real

and should expose the benchmark they're bound to via [`metric_benchmark`](@ref). `dataset` is
produced by running the policy under evaluation (e.g. via [`generate_dataset`](@ref)`(bench,
N; target_policy=policy)` for static benchmarks, or [`evaluate_policy!`](@ref) for dynamic
ones).
"""
abstract type AbstractMetric{B<:AbstractBenchmark} end

"""
$TYPEDSIGNATURES

Display name used in [`MetricStats`](@ref) printing and default plot labels.
"""
metric_name(m::AbstractMetric) = string(nameof(typeof(m)))

"""
$TYPEDSIGNATURES

Short human-readable description of what the metric computes. Defaults to an empty string.
"""
metric_description(::AbstractMetric) = ""

"""
$TYPEDSIGNATURES

Return the benchmark `m` is bound to.
"""
function metric_benchmark end

"""
$TYPEDEF

Generic ad hoc metric wrapping an arbitrary `(bench, dataset) -> Real` callable, for quick
user-defined metrics without writing a dedicated [`AbstractMetric`](@ref) subtype. Mirrors
[`Policy`](@ref)'s shape, bound to a specific `bench`.

# Fields
$TYPEDFIELDS
"""
struct Metric{B<:AbstractBenchmark,F} <: AbstractMetric{B}
    "benchmark this metric is bound to"
    bench::B
    "metric name"
    name::String
    "metric description"
    description::String
    "metric function, called as `f(bench, dataset)`"
    f::F
end

metric_name(m::Metric) = m.name
metric_description(m::Metric) = m.description
metric_benchmark(m::Metric) = m.bench

function Base.show(io::IO, m::Metric)
    println(io, "$(m.name): $(m.description)")
    return nothing
end

"""
$TYPEDSIGNATURES

Predefined [`Metric`](@ref): the objective value of the evaluated policy's decision.

For static (and [`SampleAverageApproximation`](@ref)-wrapped stochastic) benchmarks: `op`
(default `mean`) of `objective_value(bench, sample, sample.y)` over a resolved dataset, i.e.
one whose `y` field holds the decision made by the policy under evaluation (e.g. produced
via [`generate_dataset`](@ref)`(bench, N; target_policy)`). See the method added in
`static_benchmark.jl`.

For dynamic benchmarks: `op` of `sample.extra.reward` over one trajectory, as produced by
[`rollout!`](@ref)/[`evaluate_policy!`](@ref). No sign flip is applied — this mirrors the
`total_reward` already accumulated by `rollout!` exactly, so its interpretation (reward to
maximize vs. cost to minimize) follows whatever convention the benchmark's `step!` uses.
Construct with `RewardMetric(bench; op=sum)` to match `evaluate_policy!`'s per-episode total
reward. Requires each sample's `.extra` to carry a `reward` field (the default produced by
`rollout!`). See the method added in `dynamic_benchmark.jl`.
"""
function RewardMetric end

"""
$TYPEDSIGNATURES

Callable wrapped by [`RelativeGapMetric`](@ref)'s [`Metric`](@ref) — not exported, an
implementation detail.
"""
struct GapEvaluator{S,MX,F}
    statistical_model::S
    maximizer::MX
    op::F
end

function (g::GapEvaluator)(bench::AbstractBenchmark, dataset::AbstractVector{<:DataSample})
    return compute_gap(bench, dataset, g.statistical_model, g.maximizer, g.op)
end

"""
$TYPEDSIGNATURES

Predefined [`Metric`](@ref) wrapping [`compute_gap`](@ref): average relative optimality gap
of `statistical_model`+`maximizer` against the dataset's target `y`. Unlike [`RewardMetric`](@ref),
this expects the *original labeled* dataset (target `y` present, e.g. from an anticipative
`target_policy`), not a policy-resolved one — `statistical_model` and `maximizer` are applied
internally, exactly as `compute_gap` already does.
"""
function RelativeGapMetric(
    bench::AbstractBenchmark,
    statistical_model,
    maximizer;
    op=mean,
    name="relative_gap",
    description="Average relative optimality gap of statistical_model+maximizer vs. the dataset's target.",
)
    return Metric(bench, name, description, GapEvaluator(statistical_model, maximizer, op))
end

"""
    evaluate_metric(metric, dataset::AbstractVector{<:DataSample}) -> Real
    evaluate_metric(metric, datasets::AbstractVector{<:AbstractVector{<:DataSample}}) -> MetricStats

Evaluate `metric` on a single resolved dataset (returns a scalar), or on a collection of
datasets — one value per dataset, e.g. one per episode or per instance-batch — returning a
[`MetricStats`](@ref) for summary statistics and plotting. `metric` already knows which
benchmark it targets (see [`metric_benchmark`](@ref)), so `bench` is never passed here.

The multi-dataset method accepts exactly what [`evaluate_policy!`](@ref) already returns as
its `datasets` output, so the two compose directly:

```julia
rewards, datasets = evaluate_policy!(policy, envs, episodes)
stats = evaluate_metric(RewardMetric(bench; op=sum), datasets)
```
"""
function evaluate_metric(metric::Metric, dataset::AbstractVector{<:DataSample})
    return metric.f(metric.bench, dataset)
end

"""
$TYPEDSIGNATURES

Evaluate `metric` once per dataset in `datasets`, collecting the resulting values into a
[`MetricStats`](@ref). `label` identifies the policy that produced `datasets` (e.g. a
[`Policy`](@ref)'s `name`) — pass it to compare several policies later by plotting a
`Vector{<:MetricStats}` (see [`plot_metric`](@ref)).
"""
function evaluate_metric(
    metric::AbstractMetric,
    datasets::AbstractVector{<:AbstractVector{<:DataSample}},
    label::AbstractString="",
)
    return MetricStats(metric, label, Float64[evaluate_metric(metric, d) for d in datasets])
end

"""
$TYPEDSIGNATURES

Evaluate `metric` for several policies at once, one `label => datasets` pair each, returning
a `Vector{MetricStats}` ready to compare/plot together (see [`plot_metric`](@ref)):

```julia
comparison = evaluate_metric(rm, "Greedy" => datasets_greedy, "Lazy" => datasets_lazy)
plot_metric(comparison)
```
"""
function evaluate_metric(
    metric::AbstractMetric,
    policies::Pair{<:AbstractString,<:AbstractVector{<:AbstractVector{<:DataSample}}}...,
)
    return [evaluate_metric(metric, datasets, label) for (label, datasets) in policies]
end

"""
$TYPEDEF

Per-dataset values of a [`AbstractMetric`](@ref), collected across several datasets/episodes,
together with the metric that produced them. Used for summary statistics (`mean`, `std`,
`quantile`) and for plotting (see [`plot_metric`](@ref)).

`label` identifies the policy that was evaluated (e.g. a [`Policy`](@ref)'s `name`, or any
other string of your choosing). Plotting a `Vector{<:MetricStats}` (one per policy, same
metric) groups them by `label` for comparison — no dedicated "comparison" type is needed.

# Fields
$TYPEDFIELDS
"""
struct MetricStats{M<:AbstractMetric}
    "metric that produced these values"
    metric::M
    "policy label"
    label::String
    "one value per dataset"
    values::Vector{Float64}
end

"""
$TYPEDSIGNATURES

Mean of the per-dataset metric values.
"""
mean_metric(s::MetricStats) = mean(s.values)

"""
$TYPEDSIGNATURES

Standard deviation of the per-dataset metric values.
"""
std_metric(s::MetricStats) = std(s.values)

"""
$TYPEDSIGNATURES

Quantile `p` of the per-dataset metric values.
"""
quantile_metric(s::MetricStats, p) = quantile(s.values, p)

function Base.show(io::IO, s::MetricStats)
    name =
        isempty(s.label) ? metric_name(s.metric) : "$(metric_name(s.metric)) ($(s.label))"
    println(
        io, "$name: mean=$(mean_metric(s)), std=$(std_metric(s)) (n=$(length(s.values)))"
    )
    return nothing
end

"""
    plot_metric(stats::MetricStats; kwargs...)
    plot_metric(stats::AbstractVector{<:MetricStats}; kwargs...)

Plot a [`MetricStats`](@ref) (boxplot of its per-dataset values), or compare several policies
by plotting a `Vector{<:MetricStats}` (one per policy, grouped by `label` — a boxplot with one
box per policy). A generic default is provided when `StatsPlots` is loaded (`using Plots,
StatsPlots`). Custom metrics can override this via dispatch on `MetricStats{<:MyMetricType}`
for a specialised rendering.
"""
function plot_metric end
