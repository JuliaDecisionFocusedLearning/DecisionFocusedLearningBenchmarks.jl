"""
$TYPEDEF

Abstract type for evaluation metrics on **dynamic** benchmarks.
A dynamic metric is a function of a single episode: implement 
`(m::MyDynamicMetric)(episode::Vector{DataSample}) -> Real` and [`metric_benchmark`](@ref). 
[`evaluate_metric`](@ref) maps it over a vector of episodes, one value per episode. 
"""
abstract type AbstractDynamicMetric{B<:AbstractDynamicBenchmark} <: AbstractMetric{B} end

"""
$TYPEDSIGNATURES

Evaluate a dynamic metric over a vector of `episodes` (e.g. a vector of `datasets` returned by
[`evaluate_policy!`](@ref)). Returns a [`MetricStats`](@ref) with one value per episode.
`label` tags the result with the evaluated policy (see [`evaluate_metric`](@ref)).
"""
function evaluate_metric(
    m::AbstractDynamicMetric,
    episodes::AbstractVector{<:AbstractVector{<:DataSample}},
    label::AbstractString="",
)
    return MetricStats(m, label, Float64[m(episode) for episode in episodes])
end

"""
$TYPEDEF

Generic ad hoc dynamic metric wrapping a per-episode `f(bench, episode) -> Real` callable.
Built via [`Metric`](@ref)`(bench, name, description, f)` on a dynamic benchmark.

# Fields
$TYPEDFIELDS
"""
struct DynamicMetric{B<:AbstractDynamicBenchmark,F} <: AbstractDynamicMetric{B}
    "benchmark this metric is bound to"
    bench::B
    "metric name"
    name::String
    "metric description"
    description::String
    "per-episode metric function, called as `f(bench, episode)`"
    f::F
end

metric_name(m::DynamicMetric) = m.name
metric_description(m::DynamicMetric) = m.description

"""
$TYPEDSIGNATURES

Return the dynamic benchmark `m` is bound to.
"""
metric_benchmark(m::DynamicMetric) = m.bench
(m::DynamicMetric)(episode::AbstractVector{<:DataSample}) = m.f(m.bench, episode)

"""
    Metric(bench::AbstractDynamicBenchmark, name, description, f) -> DynamicMetric

Build a generic ad hoc dynamic metric bound to `bench`, wrapping a per-episode function
`f(bench, episode) -> Real`.
"""
function Metric(bench::AbstractDynamicBenchmark, name, description, f)
    return DynamicMetric(bench, name, description, f)
end

"Per-episode evaluator for [`RewardMetric`](@ref)."
struct RewardEvaluator{O}
    op::O
end

function (r::RewardEvaluator)(bench, episode::AbstractVector{<:DataSample})
    if !isempty(episode) && !haskey(first(episode).extra, :reward)
        return error(
            "RewardMetric expects each DataSample in the episode to carry a `reward` field " *
            "in `.extra` (as `rollout!` produces by default), but $(typeof(bench)) episodes " *
            "do not. Define a custom dynamic metric for this benchmark instead.",
        )
    end
    return r.op(sample.extra.reward for sample in episode)
end

"""
    RewardMetric(bench::AbstractDynamicBenchmark; op=sum) -> DynamicMetric

Predefined reward metric per-episode. Operator `op` (default `sum`) aggregates the rewards
of `sample.extra.reward`. Requires each sample's `.extra` to carry a `reward` field.
"""
function RewardMetric(bench::AbstractDynamicBenchmark; op=sum)
    return DynamicMetric(
        bench,
        "reward",
        "Total reward accumulated by the evaluated policy over an episode.",
        RewardEvaluator(op),
    )
end

"""
$TYPEDEF

Predefined relative gap metric: computes the per-episode relative gap of a test run against a stored
`target_datasets` (e.g. anticipative) run, comparing episode returns `sum(sample.extra.reward)`. 

# Fields
$TYPEDFIELDS
"""
struct DynamicGapMetric{B<:AbstractDynamicBenchmark,T} <: AbstractDynamicMetric{B}
    "benchmark this metric is bound to"
    bench::B
    "reference (target) episodes, one per evaluation scenario"
    target_datasets::T
    "name of the metric"
    name::String
    "description of the metric"
    description::String
end

metric_name(m::DynamicGapMetric) = m.name
metric_description(m::DynamicGapMetric) = m.description

"""
$TYPEDSIGNATURES

Return the dynamic benchmark `m` is bound to.
"""
metric_benchmark(m::DynamicGapMetric) = m.bench

"""
    RelativeGapMetric(bench::AbstractDynamicBenchmark, target_datasets) -> DynamicGapMetric

Dynamic [`RelativeGapMetric`](@ref): build a [`DynamicGapMetric`](@ref) from the reference
`target_datasets` (e.g. episodes of the anticipative policy from [`evaluate_policy!`](@ref)).
Evaluate it on the test policy's episodes with [`evaluate_metric`](@ref).
"""
function RelativeGapMetric(
    bench::AbstractDynamicBenchmark,
    target_datasets;
    name="relative_gap",
    description="Relative gap of the test policy against the target policy",
)
    return DynamicGapMetric(bench, target_datasets, name, description)
end

"""
$TYPEDSIGNATURES

Evaluate a [`DynamicGapMetric`](@ref) over `test_datasets`, returning a [`MetricStats`](@ref)
with one relative gap value per episode against the metric's stored `target_datasets`
(aligned by index; `target_datasets` and `test_datasets` must have the same length).
"""
function evaluate_metric(
    m::DynamicGapMetric,
    test_datasets::AbstractVector{<:AbstractVector{<:DataSample}},
    label::AbstractString="",
)
    n = length(test_datasets)
    @assert length(m.target_datasets) == n "target and test datasets must be aligned (got \
        $(length(m.target_datasets)) target vs $n test episodes)"
    values = Vector{Float64}(undef, n)
    sign = is_minimization_problem(m.bench) ? 1 : -1
    for i in 1:n
        target_return = sum(s.extra.reward for s in m.target_datasets[i])
        test_return = sum(s.extra.reward for s in test_datasets[i])
        # gap is positive : minimization => target_return < test_return, maximization => target_return > test_return
        values[i] = sign * (test_return - target_return) / abs(target_return)
    end
    return MetricStats(m, label, values)
end
