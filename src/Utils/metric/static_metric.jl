"""
$TYPEDEF

Abstract type for evaluation metrics on **static** and
[`SampleAverageApproximation`](@ref)-wrapped stochastic benchmarks.
A static metric is a function of a single sample: implement `(m::MyStaticMetric)(sample::DataSample) -> Real`
and [`metric_benchmark`](@ref). [`evaluate_metric`](@ref) maps it over the samples of a dataset,
one value per instance.
"""
abstract type AbstractStaticMetric{B<:AbstractStaticBenchmark} <: AbstractMetric{B} end

"""
$TYPEDSIGNATURES

Evaluate a static metric over a `dataset`, returning a [`MetricStats`](@ref) with one value
per sample. `label` tags the result with the evaluated policy (see [`evaluate_metric`](@ref)).
"""
function evaluate_metric(
    m::AbstractStaticMetric, dataset::AbstractVector{<:DataSample}, label::AbstractString=""
)
    return MetricStats(m, label, Float64[m(sample) for sample in dataset])
end

"""
$TYPEDEF

Generic ad hoc static metric wrapping a per-sample `f(bench, sample) -> Real` callable.
Built via [`Metric`](@ref)`(bench, name, description, f)` on a static benchmark.

# Fields
$TYPEDFIELDS
"""
struct StaticMetric{B<:AbstractStaticBenchmark,F} <: AbstractStaticMetric{B}
    "benchmark this metric is bound to"
    bench::B
    "metric name"
    name::String
    "metric description"
    description::String
    "per-sample metric function, called as `f(bench, sample)`"
    f::F
end

metric_name(m::StaticMetric) = m.name
metric_description(m::StaticMetric) = m.description

"""
$TYPEDSIGNATURES

Return the static benchmark `m` is bound to.
"""
metric_benchmark(m::StaticMetric) = m.bench
(m::StaticMetric)(sample::DataSample) = m.f(m.bench, sample)

"""
    Metric(bench::AbstractStaticBenchmark, name, description, f) -> StaticMetric

Build a generic ad hoc static metric bound to `bench`, wrapping a per-sample function
`f(bench, sample) -> Real`.
"""
function Metric(bench::AbstractStaticBenchmark, name, description, f)
    return StaticMetric(bench, name, description, f)
end

"Per-sample evaluator for [`ObjectiveMetric`](@ref) — internal."
struct ObjectiveEvaluator end
(::ObjectiveEvaluator)(bench, sample::DataSample) = objective_value(bench, sample, sample.y)

"""
    ObjectiveMetric(bench::AbstractStaticBenchmark) -> StaticMetric

Predefined static metric (a [`StaticMetric`](@ref)): per-sample
`objective_value(bench, sample, sample.y)` over a resolved dataset (with decisions `y`).
`mean(stats)` then gives the mean objective across instances.
"""
function ObjectiveMetric(bench::AbstractStaticBenchmark)
    return StaticMetric(
        bench,
        "objective",
        "Objective value of the evaluated policy's decision.",
        ObjectiveEvaluator(),
    )
end

"Per-sample evaluator for the static [`RelativeGapMetric`](@ref)."
struct StaticGapEvaluator{S,MX}
    statistical_model::S
    maximizer::MX
end

function (g::StaticGapEvaluator)(bench, sample::DataSample)
    target_obj = objective_value(bench, sample)
    θ = g.statistical_model(sample.x)
    y = g.maximizer(θ; sample.context...)
    obj = objective_value(bench, sample, y)
    Δ = is_minimization_problem(bench) ? obj - target_obj : target_obj - obj
    return Δ / abs(target_obj)
end

"""
    RelativeGapMetric(bench::AbstractStaticBenchmark, statistical_model, maximizer) -> StaticMetric

Static [`RelativeGapMetric`](@ref) (a [`StaticMetric`](@ref)): per-sample relative optimality
gap of `statistical_model`+`maximizer` against the sample's target `y`, `Δ / abs(target_obj)`
with `Δ = obj - target_obj` for minimization (reversed otherwise, per
[`is_minimization_problem`](@ref)).
notes: `mean(stats)` for this metric mirrors [`compute_gap`](@ref).
"""
function RelativeGapMetric(bench::AbstractStaticBenchmark, statistical_model, maximizer)
    return StaticMetric(
        bench,
        "relative_gap",
        "Relative optimality gap of statistical_model+maximizer vs. the sample's target.",
        StaticGapEvaluator(statistical_model, maximizer),
    )
end
