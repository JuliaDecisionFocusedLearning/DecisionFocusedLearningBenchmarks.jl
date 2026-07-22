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

"""
$TYPEDSIGNATURES

Callable wrapped by [`RewardMetric`](@ref)'s [`Metric`](@ref) for static (and
[`SampleAverageApproximation`](@ref)-wrapped stochastic) benchmarks — not exported, an
implementation detail.
"""
struct StaticRewardEvaluator{F}
    op::F
end

function (r::StaticRewardEvaluator)(bench, dataset::AbstractVector{<:DataSample})
    return r.op(objective_value(bench, sample, sample.y) for sample in dataset)
end

"""
$TYPEDSIGNATURES

[`RewardMetric`](@ref) for static (and [`SampleAverageApproximation`](@ref)-wrapped
stochastic) benchmarks: `op` (default `mean`) of `objective_value(bench, sample, sample.y)`
over a resolved dataset, i.e. one whose `y` field holds the decision made by the policy
under evaluation (e.g. produced via [`generate_dataset`](@ref)`(bench, N; target_policy)`).
"""
function RewardMetric(
    bench::AbstractStaticBenchmark;
    op=mean,
    name="reward",
    description="Objective value of the evaluated policy's decision.",
)
    return Metric(bench, name, description, StaticRewardEvaluator(op))
end
