"""
$TYPEDEF

Abstract type interface for benchmark problems.

# Mandatory methods to implement for any benchmark:
Choose one of three primary implementation strategies:
- Implement [`generate_instance`](@ref) (returns a [`DataSample`](@ref) with `y=nothing`).
  The default [`generate_sample`](@ref) forwards the call directly; [`generate_dataset`](@ref)
  applies `target_policy` afterwards if provided.
- Override [`generate_sample`](@ref) directly when the sample requires custom logic.
  [`generate_dataset`](@ref) applies `target_policy` to the result after the call returns.
- Override [`generate_dataset`](@ref) directly when samples cannot be drawn independently.

Also implement:
- [`generate_statistical_model`](@ref)
- [`generate_maximizer`](@ref)

# Optional methods (defaults provided)
- [`is_minimization_problem`](@ref): defaults to `true`
- [`objective_value`](@ref): defaults to `dot(θ, y)`
- [`compute_gap`](@ref): default implementation provided; override for custom evaluation
- [`has_visualization`](@ref): defaults to `false`

# Optional methods (no default, require `Plots` to be loaded)
- [`plot_instance`](@ref), [`plot_solution`](@ref)
- [`generate_baseline_policies`](@ref)
"""
abstract type AbstractBenchmark end

"""
    generate_instance(::AbstractBenchmark, rng::AbstractRNG; kwargs...) -> DataSample

Generate a single unlabeled [`DataSample`](@ref) (with `y=nothing`) for the benchmark.
"""
function generate_instance(bench::AbstractBenchmark, rng::AbstractRNG; kwargs...)
    return error(
        "`generate_instance` is not implemented for $(typeof(bench)). " *
        "Implement `generate_instance(::$(typeof(bench)), rng; kwargs...) -> DataSample` " *
        "or override `generate_sample` directly.",
    )
end

"""
    generate_sample(::AbstractBenchmark, rng::AbstractRNG; kwargs...) -> DataSample

Generate a single [`DataSample`](@ref) for the benchmark.

**Default** (when [`generate_instance`](@ref) is implemented):
Calls [`generate_instance`](@ref) and returns the result directly.

Override this method when sample generation requires custom logic. Labeling via
`target_policy` is always applied by [`generate_dataset`](@ref) after this call returns.
"""
function generate_sample(bench::AbstractBenchmark, rng; kwargs...)
    return generate_instance(bench, rng; kwargs...)
end

"""
    generate_dataset(::AbstractBenchmark, dataset_size::Int; target_policy=nothing, kwargs...) -> Vector{<:DataSample}

Generate a `Vector` of [`DataSample`](@ref) of length `dataset_size` for given benchmark.
Content of the dataset can be visualized using [`plot_solution`](@ref), when it applies.

By default, it uses [`generate_sample`](@ref) to create each sample in the dataset, and passes any
keyword arguments to it. `target_policy` is applied if provided, it is called on each sample
after [`generate_sample`](@ref) returns.
"""
function generate_dataset(
    bench::AbstractBenchmark,
    dataset_size::Int;
    target_policy=nothing,
    seed=nothing,
    rng=MersenneTwister(seed),
    kwargs...,
)
    Random.seed!(rng, seed)
    return [
        begin
            sample = generate_sample(bench, rng; kwargs...)
            isnothing(target_policy) ? sample : target_policy(sample)
        end for _ in 1:dataset_size
    ]
end

"""
    generate_maximizer(::AbstractBenchmark; kwargs...)

Returns a callable `f(θ; kwargs...) -> y`, solving a maximization problem.
"""
function generate_maximizer end

"""
    generate_statistical_model(::AbstractBenchmark, seed=nothing; kwargs...)

Returns an untrained statistical model (usually a Flux neural network) that maps a
feature matrix `x` to an output array `θ`. The `seed` parameter controls initialization
randomness for reproducibility.
"""
function generate_statistical_model end

"""
    generate_baseline_policies(::AbstractBenchmark) -> NamedTuple or Tuple

Return named baseline policies for the benchmark. Each policy is a callable.

- For static/stochastic benchmarks: signature `(sample) -> DataSample`.
- For dynamic benchmarks: signature `(env) -> Vector{DataSample}` (full trajectory).
"""
function generate_baseline_policies end

"""
    has_visualization(::AbstractBenchmark) -> Bool

Return `true` if `plot_instance` and `plot_solution` are implemented for this benchmark
(requires `Plots` to be loaded). Default is `false`.
"""
has_visualization(::AbstractBenchmark) = false

"""
    plot_instance(bench::AbstractBenchmark, sample::DataSample; kwargs...)

Plot the problem instance (no solution). Only available when `Plots` is loaded.
"""
function plot_instance end

"""
    plot_solution(bench::AbstractBenchmark, sample::DataSample; kwargs...)

Plot the instance with `sample.y` overlaid. Only available when `Plots` is loaded.
"""
function plot_solution end

"""
    compute_gap(::AbstractBenchmark, dataset::Vector{<:DataSample}, statistical_model, maximizer) -> Float64

Compute the average relative optimality gap of the pipeline on the dataset.
"""
function compute_gap end

"""
$TYPEDSIGNATURES

Compute `dot(θ, y)`. Override for non-linear objectives.
"""
function objective_value(::AbstractBenchmark, θ::AbstractArray, y::AbstractArray)
    return dot(θ, y)
end

"""
$TYPEDSIGNATURES

Compute the objective value of given solution `y`.
"""
function objective_value(
    bench::AbstractBenchmark, sample::DataSample{CTX,EX,F,S,C}, y::AbstractArray
) where {CTX,EX,F,S,C<:AbstractArray}
    return objective_value(bench, sample.θ, y)
end

"""
$TYPEDSIGNATURES

Compute the objective value of the target in the sample (needs to exist).
"""
function objective_value(
    bench::AbstractBenchmark, sample::DataSample{CTX,EX,F,S,C}
) where {CTX,EX,F,S<:AbstractArray,C}
    return objective_value(bench, sample, sample.y)
end

"""
$TYPEDSIGNATURES

Check if the benchmark is a minimization problem.
"""
function is_minimization_problem(::AbstractBenchmark)
    return true
end

"""
$TYPEDSIGNATURES

Default implementation of [`compute_gap`](@ref): average relative optimality gap over `dataset`.
Requires samples with `x`, `θ`, and `y` fields. Override for custom evaluation logic.
"""
function compute_gap(
    bench::AbstractBenchmark,
    dataset::AbstractVector{<:DataSample},
    statistical_model,
    maximizer,
    op=mean,
)
    check = is_minimization_problem(bench)

    return op(
        map(dataset) do sample
            target_obj = objective_value(bench, sample)
            x = sample.x
            θ = statistical_model(x)
            y = maximizer(θ; sample.instance_kwargs...)
            obj = objective_value(bench, sample, y)
            Δ = check ? obj - target_obj : target_obj - obj
            return Δ / abs(target_obj)
        end,
    )
end

"""
$TYPEDEF

Abstract type interface for single-stage stochastic benchmark problems.

A stochastic benchmark separates the problem into an **instance** (the
context known before the scenario is revealed) and a **random scenario** (the uncertain
part). Decisions are taken by seeing only the instance. Scenarios are used to generate
anticipative targets and compute objective values.

# Required methods ([`ExogenousStochasticBenchmark`](@ref) only)
- [`generate_instance`](@ref)`(bench, rng)`: returns a [`DataSample`](@ref) with instance
  and features but **no scenario**. Scenarios are added later by [`generate_dataset`](@ref)
  via [`generate_scenario`](@ref).
- [`generate_scenario`](@ref)`(bench, rng; kwargs...)`: draws a random scenario.
  Solver kwargs are spread from `sample.instance_kwargs`; context latents from `ctx.extra`.

# Optional methods
- [`generate_context`](@ref)`(bench, rng, instance_sample)`: enriches the instance with
  observable context (default: identity). Override for contextual stochastic benchmarks.
- [`generate_anticipative_solver`](@ref)`(bench)`: returns a callable
  `(scenario; kwargs...) -> y` that computes the anticipative solution per scenario.
- [`generate_parametric_anticipative_solver`](@ref)`(bench)`: returns a callable
  `(θ, scenario; kwargs...) -> y` for the parametric anticipative subproblem
  `argmin_{y ∈ Y} c(y, scenario) + θᵀy`.

# Dataset generation (exogenous only)
[`generate_dataset`](@ref) is specialised for [`ExogenousStochasticBenchmark`](@ref) and
supports all three standard structures via `nb_scenarios` and `nb_contexts`:

| Setting | Call |
|---------|------|
| 1 instance with K scenarios  | `generate_dataset(bench, 1; nb_scenarios=K)` |
| N instances with 1 scenario  | `generate_dataset(bench, N)` (default) |
| N instances with K scenarios | `generate_dataset(bench, N; nb_scenarios=K)` |
| N instances with M contexts × K scenarios | `generate_dataset(bench, N; nb_contexts=M, nb_scenarios=K)` |

By default (no `target_policy`), each [`DataSample`](@ref) has `instance_kwargs` holding
the solver kwargs and `extra=(; scenario)` holding one scenario.

Provide a `target_policy(instance_sample, ctx_sample, scenarios) -> Vector{DataSample}`
to compute labels. This covers both anticipative (K samples, one per scenario) and SAA
(1 sample from all K scenarios) labeling strategies.
"""
abstract type AbstractStochasticBenchmark{exogenous} <: AbstractBenchmark end

is_exogenous(::AbstractStochasticBenchmark{exogenous}) where {exogenous} = exogenous
is_endogenous(::AbstractStochasticBenchmark{exogenous}) where {exogenous} = !exogenous

"Alias for [`AbstractStochasticBenchmark`](@ref)`{true}`. Uncertainty is independent of decisions."
const ExogenousStochasticBenchmark = AbstractStochasticBenchmark{true}

"Alias for [`AbstractStochasticBenchmark`](@ref)`{false}`. Uncertainty depends on decisions."
const EndogenousStochasticBenchmark = AbstractStochasticBenchmark{false}

"""
    generate_scenario(::ExogenousStochasticBenchmark, rng::AbstractRNG; kwargs...) -> scenario

Draw a random scenario. Solver kwargs are passed as keyword arguments spread from
`sample.instance_kwargs`, and context latents (if any) are spread from `ctx.extra`:

    ξ = generate_scenario(bench, rng; ctx.extra..., ctx.instance_kwargs...)
"""
function generate_scenario end

"""
    generate_context(bench::AbstractStochasticBenchmark, rng, instance_sample::DataSample)
        -> DataSample

Enrich `instance_sample` with observable context drawn from `rng`.
Returns a new `DataSample` extending the instance: `.x` is the final ML feature vector
(possibly augmented with context features) and `.extra` holds any latent context fields
needed by [`generate_scenario`](@ref).

**Default**: returns `instance_sample` unchanged — no context augmentation.
Non-contextual benchmarks (e.g. SVS) use this default.

**Override** to add per-sample context features:
```julia
function generate_context(bench::MyBench, rng, instance_sample::DataSample)
    x_raw = randn(rng, Float32, bench.d)
    return DataSample(;
        x=vcat(instance_sample.x, x_raw),
        instance_sample.instance_kwargs...,
        extra=(; x_raw),
    )
end
```
Fields in `.extra` are spread into [`generate_scenario`](@ref) as kwargs.
"""
function generate_context(::AbstractStochasticBenchmark, rng, instance_sample::DataSample)
    return instance_sample
end

"""
    generate_anticipative_solver(::AbstractBenchmark) -> callable

Return a callable that computes the anticipative solution.

- For [`AbstractStochasticBenchmark`](@ref): returns `(scenario; context...) -> y`.
- For [`AbstractDynamicBenchmark`](@ref): returns
  `(env; reset_env=true, kwargs...) -> Vector{DataSample}`, a full training trajectory.
  `reset_env=true` resets the env before solving (initial dataset building);
  `reset_env=false` starts from the current env state.
"""
function generate_anticipative_solver end

"""
    generate_parametric_anticipative_solver(::ExogenousStochasticBenchmark) -> callable

**Optional.** Return a callable `(θ, scenario; kwargs...) -> y` that solves the
parametric anticipative subproblem:

    argmin_{y ∈ Y(instance)}  c(y, scenario) + θᵀy
"""
function generate_parametric_anticipative_solver end

"""
$TYPEDSIGNATURES

Default [`generate_sample`](@ref) for exogenous stochastic benchmarks.

Calls [`generate_instance`](@ref), then [`generate_context`](@ref) (default: identity),
draws scenarios via [`generate_scenario`](@ref), then:
- Without `target_policy`: returns M×K unlabeled samples (`nb_contexts` contexts ×
  `nb_scenarios` scenarios each), each with one scenario in `extra=(; scenario=ξ)`.
- With `target_policy`: calls `target_policy(instance_sample, ctx_sample, scenarios)`
  per context and returns the result.

`target_policy(instance_sample, ctx_sample, scenarios) -> Vector{DataSample}` enables
anticipative labeling (K samples, one per scenario) or SAA (1 sample aggregating all K
scenarios).
"""
function generate_sample(
    bench::ExogenousStochasticBenchmark,
    rng;
    target_policy=nothing,
    nb_scenarios::Int=1,
    nb_contexts::Int=1,
    kwargs...,
)
    instance_sample = generate_instance(bench, rng; kwargs...)
    result = DataSample[]
    for _ in 1:nb_contexts
        ctx = generate_context(bench, rng, instance_sample)
        if isnothing(target_policy)
            for _ in 1:nb_scenarios
                ξ = generate_scenario(bench, rng; ctx.extra..., ctx.instance_kwargs...)
                push!(
                    result,
                    DataSample(;
                        x=ctx.x,
                        θ=ctx.θ,
                        ctx.instance_kwargs...,
                        extra=(; ctx.extra..., scenario=ξ),
                    ),
                )
            end
        else
            scenarios = [
                generate_scenario(bench, rng; ctx.extra..., ctx.instance_kwargs...) for
                _ in 1:nb_scenarios
            ]
            append!(result, target_policy(instance_sample, ctx, scenarios))
        end
    end
    return result
end

"""
$TYPEDSIGNATURES

Specialised [`generate_dataset`](@ref) for exogenous stochastic benchmarks.

Generates `nb_instances` problem instances, each with `nb_contexts` context draws
and `nb_scenarios` scenario draws per context. The scenario→sample mapping is controlled
by the `target_policy`:
- Without `target_policy` (default): M contexts × K scenarios produce M×K unlabeled
  samples per instance.
- With `target_policy(instance_sample, ctx_sample, scenarios) -> Vector{DataSample}`:
  enables anticipative labeling (K labeled samples) or SAA (1 sample aggregating all K
  scenarios).

# Keyword arguments
- `nb_scenarios::Int = 1`: scenarios per context (K).
- `nb_contexts::Int = 1`: context draws per instance (M).
- `target_policy`: when provided, called as
  `target_policy(instance_sample, ctx_sample, scenarios)` to compute labels.
  Defaults to `nothing` (unlabeled samples).
- `seed`: passed to `MersenneTwister` when `rng` is not provided.
- `rng`: random number generator; overrides `seed` when provided.
- `kwargs...`: forwarded to [`generate_sample`](@ref).
"""
function generate_dataset(
    bench::ExogenousStochasticBenchmark,
    nb_instances::Int;
    target_policy=nothing,
    nb_scenarios::Int=1,
    nb_contexts::Int=1,
    seed=nothing,
    rng=MersenneTwister(seed),
    kwargs...,
)
    Random.seed!(rng, seed)
    samples = DataSample[]
    for _ in 1:nb_instances
        new_samples = generate_sample(
            bench, rng; target_policy, nb_scenarios, nb_contexts, kwargs...
        )
        append!(samples, new_samples)
    end
    return samples
end

"""
$TYPEDEF

Transforms an [`ExogenousStochasticBenchmark`](@ref) into a static benchmark via
Sample Average Approximation (SAA).

For each (instance, context) pair, draws `nb_scenarios` fixed scenarios. These are stored
in the sample and used for feature computation, target labeling (via `target_policy`),
and gap evaluation.

# Fields
$TYPEDFIELDS
"""
struct SAA{B<:ExogenousStochasticBenchmark} <: AbstractBenchmark
    "inner stochastic benchmark"
    benchmark::B
    "number of scenarios to draw per (instance, context) pair"
    nb_scenarios::Int
end

is_minimization_problem(saa::SAA) = is_minimization_problem(saa.benchmark)
generate_maximizer(saa::SAA) = generate_maximizer(saa.benchmark)
function generate_statistical_model(saa::SAA; kwargs...)
    return generate_statistical_model(saa.benchmark; kwargs...)
end

function generate_sample(saa::SAA, rng; target_policy=nothing, kwargs...)
    inner = saa.benchmark
    instance_sample = generate_instance(inner, rng; kwargs...)
    ctx = generate_context(inner, rng, instance_sample)
    scenarios = [
        generate_scenario(inner, rng; ctx.extra..., ctx.instance_kwargs...) for
        _ in 1:(saa.nb_scenarios)
    ]
    if isnothing(target_policy)
        return [
            DataSample(; x=ctx.x, ctx.instance_kwargs..., extra=(; ctx.extra..., scenarios))
        ]
    else
        return target_policy(instance_sample, ctx, scenarios)
    end
end

"""
$TYPEDSIGNATURES

Specialised [`generate_dataset`](@ref) for [`SAA`](@ref).

- Without `target_policy`: returns one static [`DataSample`](@ref) per instance, with
  `nb_scenarios` stored in `extra.scenarios`.
- With `target_policy(instance_sample, ctx_sample, scenarios) -> Vector{DataSample}`:
  labels each instance using all stored scenarios (same signature as
  [`ExogenousStochasticBenchmark`](@ref) policies).
"""
function generate_dataset(
    saa::SAA,
    nb_instances::Int;
    target_policy=nothing,
    seed=nothing,
    rng=MersenneTwister(seed),
    kwargs...,
)
    Random.seed!(rng, seed)
    samples = DataSample[]
    for _ in 1:nb_instances
        append!(samples, generate_sample(saa, rng; target_policy, kwargs...))
    end
    return samples
end

"""
$TYPEDSIGNATURES

Evaluate a decision `y` against stored scenarios (average over scenarios).
"""
function objective_value(saa::SAA, sample::DataSample, y::AbstractArray)
    return mean(objective_value(saa.benchmark, ξ, y) for ξ in sample.extra.scenarios)
end

"""
$TYPEDSIGNATURES

Evaluate the target solution in the sample against stored scenarios.
"""
function objective_value(
    saa::SAA, sample::DataSample{CTX,EX,F,S,C}
) where {CTX,EX,F,S<:AbstractArray,C}
    return objective_value(saa, sample, sample.y)
end

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
abstract type AbstractDynamicBenchmark{exogenous} <: AbstractStochasticBenchmark{exogenous} end

"Alias for [`AbstractDynamicBenchmark`](@ref)`{true}`. Uncertainty is independent of decisions."
const ExogenousDynamicBenchmark = AbstractDynamicBenchmark{true}

"Alias for [`AbstractDynamicBenchmark`](@ref)`{false}`. Uncertainty depends on decisions."
const EndogenousDynamicBenchmark = AbstractDynamicBenchmark{false}

"""
    generate_environment(::AbstractDynamicBenchmark, rng::AbstractRNG; kwargs...)

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
    Random.seed!(rng, seed)
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
    bench::ExogenousDynamicBenchmark,
    environments::AbstractVector;
    target_policy,
    seed=nothing,
    rng=MersenneTwister(seed),
    kwargs...,
)
    Random.seed!(rng, seed)
    samples = DataSample[]
    for env in environments
        trajectory = target_policy(env)
        append!(samples, trajectory)
    end
    return samples
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
