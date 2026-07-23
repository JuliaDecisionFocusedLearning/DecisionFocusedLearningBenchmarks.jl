"""
$TYPEDEF

Abstract type for evaluation metrics, used to evaluate and compare [`Policy`](@ref)s
on a benchmark. Parametrized by the benchmark type `B` the metric is bound to, 
this makes sense since an evaluation process is always benchmark-specific.

The metrics logic is splitted between static and dynamic benchmark:
- [`AbstractStaticMetric`](@ref): a metric is a function of **one sample**, `f(bench, sample)
  -> Real`. [`evaluate_metric`](@ref) maps it over the samples of a dataset — one value per
  instance.
- [`AbstractDynamicMetric`](@ref): a metric is a function of **one episode**, `f(bench,
  episode) -> Real` (`episode::Vector{DataSample}`, a trajectory). [`evaluate_metric`](@ref)
  maps it over episodes — one value per episode.

In both cases [`evaluate_metric`](@ref) returns a [`MetricStats`](@ref) — the distribution of
per-unit values — over which `mean_metric`/`std_metric`/`quantile_metric` give summary stats
and [`plot_metric`](@ref) plots the distribution.
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
    Metric(bench, name, description, f) -> StaticMetric | DynamicMetric

Build a generic ad hoc metric bound to `bench`, wrapping a per-unit function `f`:
- static benchmark: `f(bench, sample) -> Real` (see [`StaticMetric`](@ref)),
- dynamic benchmark: `f(bench, episode) -> Real` (see [`DynamicMetric`](@ref)).

Methods are added in `static_benchmark.jl` / `dynamic_benchmark.jl`.
"""
function Metric end

"""
    ObjectiveMetric(bench::AbstractStaticBenchmark) -> ObjectiveMetric

Predefined static metric: per-sample `objective_value(bench, sample, sample.y)`. See
`static_benchmark.jl`.
"""
function ObjectiveMetric end

"""
    RewardMetric(bench::AbstractDynamicBenchmark; op=sum) -> RewardMetric

Predefined dynamic metric: per-episode `op` (default `sum`, i.e. the return) of
`sample.extra.reward`. See `dynamic_benchmark.jl`.
"""
function RewardMetric end

"""
    RelativeGapMetric(bench::AbstractStaticBenchmark, statistical_model, maximizer) -> StaticGapMetric
    RelativeGapMetric(bench::AbstractDynamicBenchmark, target_datasets) -> DynamicGapMetric

Predefined relative optimality/target gap metric.

- **static**: per-sample gap of `statistical_model`+`maximizer` against the sample's target `y`. 
- **dynamic**: per-episode gap of a test run against a reference (e.g. anticipative) run
"""
function RelativeGapMetric end

"""
$TYPEDEF

Per-unit (sample or episode) values of an [`AbstractMetric`](@ref). Bound to the metric (field metric) and 
the policy (field label) that produced them.
This is used for summary statistics (`mean_metric`, `std_metric`,..) and for plotting.
Plotting a `Vector{<:MetricStats}` (one per policy, same metric) groups them by `label` for comparison.

Note: after the mergin the PR of policies, we could consider storing the actual policy object.

# Fields
$TYPEDFIELDS
"""
struct MetricStats{M<:AbstractMetric}
    "metric that produced these values"
    metric::M
    "policy label"
    label::String
    "one value per observation unit (sample or episode)"
    values::Vector{Float64}
end

"""
$TYPEDSIGNATURES

Mean of the per-unit metric values.
"""
mean_metric(s::MetricStats) = mean(s.values)

"""
$TYPEDSIGNATURES

Standard deviation of the per-unit metric values.
"""
std_metric(s::MetricStats) = std(s.values)

"""
$TYPEDSIGNATURES

Quantile `p` of the per-unit metric values.
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
    evaluate_metric(metric, collection, label="") -> MetricStats

Evaluate `metric` over a `collection` of observation units, returning a [`MetricStats`](@ref)
holding one value per unit. 
- **static** metric: `collection` is a dataset (`Vector{DataSample}`); one value per sample.
  E.g. `collection = generate_dataset(bench, N; target_policy=policy)`.
- **dynamic** metric: `collection` is a vector of episodes (`Vector{Vector{DataSample}}`);
  one value per episode. 
  E.g. `_, collection = evaluate_policy!(policy, env, episodes)`.

"""
function evaluate_metric end

"""
$TYPEDSIGNATURES

Evaluate `metric` for several policies at once, one `label => collection` pair each,
returning a `Vector{MetricStats}` ready to compare/plot together (see [`plot_metric`](@ref)):

```julia
comparison = evaluate_metric(rm, "Greedy" => datasets_greedy, "Lazy" => datasets_lazy)
plot_metric(comparison)
```
"""
function evaluate_metric(metric::AbstractMetric, policies::Pair{<:AbstractString}...)
    return [evaluate_metric(metric, collection, label) for (label, collection) in policies]
end

"""
    plot_metric(stats::MetricStats; kwargs...)
    plot_metric(stats::AbstractVector{<:MetricStats}; kwargs...)

Plot one distribution [`MetricStats`](@ref) or a `Vector{<:MetricStats}` for comparing policies.
It produces a boxplot of per-unit values for each `label`.
A default is provided which can be customed via dispatch on `MetricStats{<:MyMetricType}`.
"""
function plot_metric end
