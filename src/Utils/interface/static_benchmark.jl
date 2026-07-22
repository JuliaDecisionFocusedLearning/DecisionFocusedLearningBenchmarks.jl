"""
$TYPEDEF

Abstract type interface for static benchmark problems.

# Mandatory methods to implement for any static benchmark:
Choose one of three primary implementation strategies:
- Implement [`generate_instance`](@ref) (returns a [`DataSample`](@ref) with `y=nothing`).
  The default [`generate_sample`](@ref) forwards the call directly; [`generate_dataset`](@ref)
  applies `target_policy` afterwards if provided.
- Override [`generate_sample`](@ref) directly when the sample requires custom logic
  that cannot be expressed via [`generate_instance`](@ref). Applies to static benchmarks
  only, stochastic benchmarks should implement the finer-grained hooks instead
  ([`generate_instance`](@ref), [`generate_context`](@ref), [`generate_scenario`](@ref)).
  [`generate_dataset`](@ref) applies `target_policy` to the result after the call returns.
- Override [`generate_dataset`](@ref) directly when samples cannot be drawn independently.

Also implement:
- [`generate_statistical_model`](@ref)
- [`generate_maximizer`](@ref)

# Optional methods (defaults provided)
- [`is_minimization_problem`](@ref): defaults to `true`
- [`compute_gap`](@ref): default implementation provided; override for custom evaluation
- [`has_visualization`](@ref): defaults to `false`

# Mandatory methods (no default)
- [`objective_value`](@ref)`(bench, sample, y)`: must be implemented by every static benchmark

# Optional methods (no default, require `Plots` to be loaded)
- [`plot_context`](@ref), [`plot_sample`](@ref)
- [`generate_baseline_policies`](@ref)
"""
abstract type AbstractStaticBenchmark <: AbstractBenchmark end

"""
    generate_sample(::AbstractStaticBenchmark, rng::AbstractRNG; kwargs...) -> DataSample

Generate a single [`DataSample`](@ref) for the benchmark.

**Default** (when [`generate_instance`](@ref) is implemented):
Calls [`generate_instance`](@ref) and returns the result directly.

Override this method when sample generation requires custom logic. Labeling via
`target_policy` is always applied by [`generate_dataset`](@ref) after this call returns.

!!! note
    This is an internal hook called by [`generate_dataset`](@ref). Prefer calling
    [`generate_dataset`](@ref) rather than this method directly.
"""
function generate_sample(bench::AbstractStaticBenchmark, rng; kwargs...)
    return generate_instance(bench, rng; kwargs...)
end

"""
    generate_dataset(::AbstractStaticBenchmark, dataset_size::Int; target_policy=nothing, kwargs...) -> Vector{<:DataSample}

Generate a `Vector` of [`DataSample`](@ref) of length `dataset_size` for given benchmark.
Content of the dataset can be visualized using [`plot_sample`](@ref), when it applies.

By default, it uses [`generate_sample`](@ref) to create each sample in the dataset, and passes any
keyword arguments to it. `target_policy` is applied if provided, it is called on each sample
after [`generate_sample`](@ref) returns.
"""
function generate_dataset(
    bench::AbstractStaticBenchmark,
    dataset_size::Int;
    target_policy=nothing,
    seed=nothing,
    rng=Xoshiro(seed),
    kwargs...,
)
    return [
        begin
            sample = generate_sample(bench, rng; kwargs...)
            isnothing(target_policy) ? sample : target_policy(sample)
        end for _ in 1:dataset_size
    ]
end

# =========================================================================================
# Static metrics
# =========================================================================================

"""
$TYPEDEF

Abstract supertype for evaluation metrics on **static** (and
[`SampleAverageApproximation`](@ref)-wrapped stochastic) benchmarks. A static metric is a
function of a single sample: implement `(m::MyStaticMetric)(sample::DataSample) -> Real` and
[`metric_benchmark`](@ref). [`evaluate_metric`](@ref) maps it over the samples of a dataset,
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
metric_benchmark(m::StaticMetric) = m.bench
(m::StaticMetric)(sample::DataSample) = m.f(m.bench, sample)

function Metric(bench::AbstractStaticBenchmark, name, description, f)
    return StaticMetric(bench, name, description, f)
end

"Per-sample evaluator for [`ObjectiveMetric`](@ref) — internal."
struct ObjectiveEvaluator end
(::ObjectiveEvaluator)(bench, sample::DataSample) = objective_value(bench, sample, sample.y)

"""
$TYPEDSIGNATURES

Predefined static metric (a [`StaticMetric`](@ref)): per-sample
`objective_value(bench, sample, sample.y)` over a resolved dataset (one whose `y` holds the
evaluated policy's decision). `mean_metric` then gives the mean objective across instances.
"""
function ObjectiveMetric(bench::AbstractStaticBenchmark)
    return StaticMetric(
        bench,
        "objective",
        "Objective value of the evaluated policy's decision.",
        ObjectiveEvaluator(),
    )
end

"Per-sample evaluator for the static [`RelativeGapMetric`](@ref) — internal."
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
$TYPEDSIGNATURES

Static [`RelativeGapMetric`](@ref) (a [`StaticMetric`](@ref)): per-sample relative optimality
gap of `statistical_model`+`maximizer` against the sample's target `y`, `Δ / abs(target_obj)`
with `Δ = obj - target_obj` for minimization (reversed otherwise, per
[`is_minimization_problem`](@ref)). `mean_metric` reproduces [`compute_gap`](@ref).
"""
function RelativeGapMetric(bench::AbstractStaticBenchmark, statistical_model, maximizer)
    return StaticMetric(
        bench,
        "relative_gap",
        "Relative optimality gap of statistical_model+maximizer vs. the sample's target.",
        StaticGapEvaluator(statistical_model, maximizer),
    )
end
