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
$TYPEDSIGNATURES

Callable wrapped by [`RewardMetric`](@ref)'s [`Metric`](@ref) for dynamic benchmarks — not
exported, an implementation detail.
"""
struct DynamicRewardEvaluator{F}
    op::F
end

function (r::DynamicRewardEvaluator)(bench, trajectory::AbstractVector{<:DataSample})
    if !isempty(trajectory) && !haskey(first(trajectory).extra, :reward)
        return error(
            "RewardMetric expects each DataSample in the trajectory to carry a `reward` " *
            "field in `.extra` (as `rollout!` produces by default), but $(typeof(bench)) " *
            "trajectories do not. Define a custom `AbstractMetric` for this benchmark instead.",
        )
    end
    return r.op(sample.extra.reward for sample in trajectory)
end

"""
$TYPEDSIGNATURES

[`RewardMetric`](@ref) for dynamic benchmarks: `op` (default `mean`) of `sample.extra.reward`
over one trajectory, as produced by [`rollout!`](@ref)/[`evaluate_policy!`](@ref). No sign
flip is applied — this mirrors the `total_reward` already accumulated by `rollout!` exactly.
Construct with `RewardMetric(bench; op=sum)` to match `evaluate_policy!`'s per-episode total
reward. Benchmarks that override `rollout!`/`step!` without storing a reward in `.extra` are
not supported by this default — define a custom [`AbstractMetric`](@ref) instead.
"""
function RewardMetric(
    bench::AbstractDynamicBenchmark;
    op=mean,
    name="reward",
    description="Objective value of the evaluated policy's decision.",
)
    return Metric(bench, name, description, DynamicRewardEvaluator(op))
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
