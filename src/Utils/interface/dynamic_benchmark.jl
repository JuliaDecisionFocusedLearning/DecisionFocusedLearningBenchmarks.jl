"""
$TYPEDEF

Abstract type interface for multi-stage stochastic (dynamic) benchmark problems.
The `{exogenous}` parameter has the same meaning (whether uncertainty is independent
of decisions) as in [`AbstractStochasticBenchmark`](@ref).

# Primary entry point
- [`build_environment`](@ref)`(bench, rng)`: mandatory hook returning a single bare environment.
  The framework wraps it in a [`SeededEnvironment`](@ref). Override [`generate_environments`](@ref)
  instead when environments cannot be drawn independently.

# Additional optional methods
- [`build_environment`](@ref)`(bench, rng)`: build a single bare environment.
  Must return an [`AbstractEnvironment`](@ref) (see `environment.jl` for the full protocol:
  [`reset!`](@ref), [`observe`](@ref), [`step!`](@ref), [`is_terminated`](@ref)). The
  environment must not manage its own seed/rng: draw randomness from the passed `rng`.
  Users obtain wrapped environments via [`generate_environment`](@ref) (one) or
  [`generate_environments`](@ref) (many).
- [`generate_baseline_policies`](@ref)`(bench)`: returns named baseline callables of
  signature `(env) -> Vector{DataSample}` (full trajectory rollout).
- [`generate_anticipative_solver`](@ref)`(bench)`: returns a callable
  `(env; reset_env=true, kwargs...) -> Vector{DataSample}` that runs the anticipative solver over a full episode. `reset_env=true` resets the environment
  before solving. `reset_env=false` starts from the current state.
- [`generate_dataset`](@ref)`(bench, environments; target_policy, ...)`: generates
  training-ready [`DataSample`](@ref)s by calling `target_policy(env)` for each environment.
  Requires `target_policy` as a mandatory keyword argument.

# Optional visualization methods (require `Plots` to be loaded)
- [`plot_trajectory`](@ref)`(bench, traj)`: plot a full episode as a grid of subplots.
- [`animate_trajectory`](@ref)`(bench, traj)`: animate a full episode.
"""
abstract type AbstractDynamicBenchmark{exogenous} <: AbstractBenchmark end

is_exogenous(::AbstractDynamicBenchmark{exogenous}) where {exogenous} = exogenous
is_endogenous(::AbstractDynamicBenchmark{exogenous}) where {exogenous} = !exogenous

"""
$TYPEDSIGNATURES

Intercepts accidental calls to the default `compute_gap` on dynamic benchmarks and throws a
descriptive error. Dynamic benchmarks do not have a generic single-sample gap computation;
override `compute_gap` directly on the concrete type if needed.
"""
function compute_gap(bench::AbstractDynamicBenchmark, args...; kwargs...)
    return error(
        "`compute_gap` is not supported for dynamic benchmarks ($(typeof(bench))). " *
        "Override `compute_gap` on the concrete type with trajectory-based evaluation logic.",
    )
end

"Alias for [`AbstractDynamicBenchmark`](@ref)`{true}`. Uncertainty is independent of decisions."
const ExogenousDynamicBenchmark = AbstractDynamicBenchmark{true}

"Alias for [`AbstractDynamicBenchmark`](@ref)`{false}`. Uncertainty depends on decisions."
const EndogenousDynamicBenchmark = AbstractDynamicBenchmark{false}

"""
    build_environment(::AbstractDynamicBenchmark, rng::AbstractRNG; kwargs...) -> AbstractEnvironment

Build a single environment for the given dynamic benchmark, drawing any
randomness from `rng`. This is the implementation hook benchmark authors provide: the
framework wraps the result in a [`SeededEnvironment`](@ref) (see [`generate_environment`](@ref)
and [`generate_environments`](@ref)), so the returned environment must not manage its own
seed or rng.

Implement this for benchmarks whose environments are drawn independently. When
environments cannot be drawn independently (e.g. loaded from files), override
[`generate_environments`](@ref) instead.
"""
function build_environment end

"""
$TYPEDSIGNATURES

Generate a single environment wrapped in a [`SeededEnvironment`](@ref), seeded from `seed`.
Equivalent to `generate_environments(bench, 1; seed)[1]`.

This should **not** be overridden: customize [`build_environment`](@ref) (independent environments)
or [`generate_environments`](@ref) (non-independent environments) instead.
"""
function generate_environment(bench::AbstractDynamicBenchmark; seed=nothing, kwargs...)
    return only(generate_environments(bench, 1; seed, kwargs...))
end

"""
$TYPEDSIGNATURES

Generate `n` environments for the given dynamic benchmark, each wrapped in a
[`SeededEnvironment`](@ref). Primary entry point for dynamic training algorithms.

The default implementation calls [`build_environment`](@ref) `n` times and wraps each
result. Override this method when environments cannot be drawn independently (e.g. loading
from files); an override must return already-wrapped [`SeededEnvironment`](@ref)s.
"""
function generate_environments(
    bench::AbstractDynamicBenchmark, n::Int; seed=nothing, kwargs...
)
    root_rng = Xoshiro(seed)
    gen_rng = Xoshiro(rand(root_rng, UInt))
    seed_rng = Xoshiro(rand(root_rng, UInt))
    return [
        SeededEnvironment(
            build_environment(bench, gen_rng; kwargs...); seed=rand(seed_rng, UInt)
        ) for _ in 1:n
    ]
end

"""
$TYPEDSIGNATURES

Generate a training dataset from pre-built environments for an exogenous dynamic benchmark.

For each environment, calls `target_policy(env)` to obtain a training trajectory
(`Vector{DataSample}`). The trajectories are concatenated into a flat dataset.

`target_policy` is a **required** keyword argument. Use [`generate_baseline_policies`](@ref)
to obtain standard baseline callables (e.g. the anticipative solver).

# Keyword arguments
- `target_policy`: **required** callable `(env) -> Vector{DataSample}`.
"""
function generate_dataset(
    bench::ExogenousDynamicBenchmark, environments::AbstractVector; target_policy, kwargs...
)
    isempty(environments) && return DataSample[]
    return reduce(vcat, (target_policy(env) for env in environments))
end

"""
$TYPEDSIGNATURES

Convenience wrapper for exogenous dynamic benchmarks: generates `n` environments
via [`generate_environments`](@ref), then calls
[`generate_dataset`](@ref)`(bench, environments; target_policy, ...)`.

`target_policy` is a **required** keyword argument.
"""
function generate_dataset(
    bench::ExogenousDynamicBenchmark, n::Int; target_policy, seed=nothing, kwargs...
)
    environments = generate_environments(bench, n; seed)
    return generate_dataset(bench, environments; target_policy, kwargs...)
end

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
metric_benchmark(m::DynamicMetric) = m.bench
(m::DynamicMetric)(episode::AbstractVector{<:DataSample}) = m.f(m.bench, episode)

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
$TYPEDSIGNATURES

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
metric_benchmark(m::DynamicGapMetric) = m.bench

"""
$TYPEDSIGNATURES

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

"""
    plot_trajectory(bench::AbstractDynamicBenchmark, trajectory::Vector{<:DataSample}; kwargs...)

Plot a full dynamic episode as a grid of state/decision subplots.
Only available when `Plots` is loaded.
"""
function plot_trajectory end

"""
    animate_trajectory(bench::AbstractDynamicBenchmark, trajectory::Vector{<:DataSample}; kwargs...)

Animate a full dynamic episode. Returns a `Plots.Animation` object
(save with `gif(result, filename)`). Only available when `Plots` is loaded.
"""
function animate_trajectory end
