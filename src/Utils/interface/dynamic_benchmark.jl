"""
$TYPEDEF

Abstract type interface for multi-stage stochastic (dynamic) benchmark problems.

Extends [`AbstractStochasticBenchmark`](@ref). The `{exogenous}` parameter retains its
meaning (whether uncertainty is independent of decisions).

# Primary entry point
- [`generate_environments`](@ref)`(bench, n; rng)`: mandatory (or implement
  [`generate_environment`](@ref)`(bench, rng)`). The count-based default calls
  [`generate_environment`](@ref) once per environment.

# Additional optional methods
- [`generate_environment`](@ref)`(bench, rng)`: initialize a single rollout environment.
  Must return an [`AbstractEnvironment`](@ref) (see `environment.jl` for the full protocol:
  [`reset!`](@ref), [`observe`](@ref), [`step!`](@ref), [`is_terminated`](@ref)).
  Implement this instead of overriding [`generate_environments`](@ref) when environments
  can be drawn independently.
- [`generate_baseline_policies`](@ref)`(bench)`: returns named baseline callables of
  signature `(env) -> Vector{DataSample}` (full trajectory rollout).
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

Intercepts accidental calls to `generate_sample` on dynamic benchmarks and throws a
descriptive error pointing at the correct entry point.
"""
function generate_sample(bench::AbstractDynamicBenchmark, rng; kwargs...)
    return error(
        "`generate_sample` is not supported for dynamic benchmarks ($(typeof(bench))). " *
        "Use `generate_environments` and " *
        "`generate_dataset(bench, environments; target_policy=...)` instead.",
    )
end

"Alias for [`AbstractDynamicBenchmark`](@ref)`{true}`. Uncertainty is independent of decisions."
const ExogenousDynamicBenchmark = AbstractDynamicBenchmark{true}

"Alias for [`AbstractDynamicBenchmark`](@ref)`{false}`. Uncertainty depends on decisions."
const EndogenousDynamicBenchmark = AbstractDynamicBenchmark{false}

"""
    generate_environment(::AbstractDynamicBenchmark, rng::AbstractRNG; kwargs...) -> AbstractEnvironment

Initialize a single environment for the given dynamic benchmark.
Primary implementation target for the count-based [`generate_environments`](@ref) default.
Override [`generate_environments`](@ref) directly when environments cannot be drawn
independently (e.g. loading from files).
"""
function generate_environment end

"""
$TYPEDSIGNATURES

Generate `n` environments for the given dynamic benchmark.
Primary entry point for dynamic training algorithms.
Override when environments cannot be drawn independently (e.g. loading from files).
"""
function generate_environments(
    bench::AbstractDynamicBenchmark,
    n::Int;
    seed=nothing,
    rng=MersenneTwister(seed),
    kwargs...,
)
    return [generate_environment(bench, rng; kwargs...) for _ in 1:n]
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
- `seed`: passed to `MersenneTwister` when `rng` is not provided.
- `rng`: random number generator.
"""
function generate_dataset(
    bench::ExogenousDynamicBenchmark, environments::AbstractVector; target_policy, kwargs...
)
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
    return generate_dataset(bench, environments; target_policy, seed, kwargs...)
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
