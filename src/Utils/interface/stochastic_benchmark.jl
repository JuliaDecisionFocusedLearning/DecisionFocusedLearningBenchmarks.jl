"""
$TYPEDEF

Abstract type interface for single-stage stochastic benchmark problems.

A stochastic benchmark separates the problem into an **instance** (the
context known before the scenario is revealed) and a **random scenario** (the uncertain
part). Decisions are taken by seeing only the instance. Scenarios are used to generate
anticipative targets and compute objective values.

# Required methods ([`ExogenousStochasticBenchmark`](@ref) only)
- [`generate_instance`](@ref)`(bench, rng)`: returns a [`DataSample`](@ref) with the
  problem instance (solver kwargs) and, if not overriding [`generate_context`](@ref),
  the ML features `x`. Scenarios are added later by [`generate_dataset`](@ref) via
  [`generate_scenario`](@ref). When [`generate_context`](@ref) is overridden, `x` may
  be absent here and constructed there instead.
- [`generate_scenario`](@ref)`(bench, rng; kwargs...)`: draws a random scenario.
  Solver kwargs are spread from `ctx.context`.

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
supports all three standard structures via `nb_scenarios` and `contexts_per_instance`:

| Setting | Call |
|---------|------|
| 1 instance with K scenarios  | `generate_dataset(bench, 1; nb_scenarios=K)` |
| N instances with 1 scenario  | `generate_dataset(bench, N)` (default) |
| N instances with K scenarios | `generate_dataset(bench, N; nb_scenarios=K)` |
| N instances with M contexts × K scenarios | `generate_dataset(bench, N; contexts_per_instance=M, nb_scenarios=K)` |

By default (no `target_policy`), each [`DataSample`](@ref) has `context` holding
the solver kwargs and `extra=(; scenario)` holding one scenario.

Provide a `target_policy(ctx_sample, scenarios) -> Vector{DataSample}`
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
`sample.context`:

    ξ = generate_scenario(bench, rng; ctx.context...)
"""
function generate_scenario end

"""
    generate_context(bench::AbstractStochasticBenchmark, rng, instance_sample::DataSample)
        -> DataSample

Enrich `instance_sample` with observable context drawn from `rng`.
Returns a new `DataSample` extending the instance: `.x` is the final ML feature vector
(possibly augmented with context features). Any latent fields needed by
[`generate_scenario`](@ref) must go into `.context` (they are spread as kwargs via
`ctx.context...`), not into `.extra`.

**Default**: returns `instance_sample` unchanged — no context augmentation.
Non-contextual benchmarks (e.g. SVS) use this default.

**Override** to add per-sample context features:
```julia
function generate_context(bench::MyBench, rng, instance_sample::DataSample)
    x_raw = randn(rng, Float32, bench.d)
    return DataSample(;
        x=vcat(instance_sample.x, x_raw),
        instance_sample.context...,
        x_raw,
    )
end
```
Fields in `.context` are spread into [`generate_scenario`](@ref) as kwargs.
"""
function generate_context(::AbstractStochasticBenchmark, rng, instance_sample::DataSample)
    return instance_sample
end

"""
    generate_anticipative_solver(::AbstractBenchmark) -> callable

Return a callable that computes the anticipative (oracle) solution.
The calling convention differs by benchmark category:

**Stochastic benchmarks** ([`AbstractStochasticBenchmark`](@ref)):
Returns `(scenario; context...) -> y`.
Called once per scenario to obtain the optimal label.

**Dynamic benchmarks** ([`AbstractDynamicBenchmark`](@ref)):
Returns `(env; reset_env=true, kwargs...) -> Vector{DataSample}`, a full trajectory.
`reset_env=true` resets the environment before solving (used for initial dataset building);
`reset_env=false` starts from the current environment state (used inside DAgger rollouts).
"""
function generate_anticipative_solver end

"""
    objective_value(::ExogenousStochasticBenchmark, sample::DataSample, y, scenario) -> Real

Compute the objective value of solution `y` for a given `scenario`.
Must be implemented by each concrete [`ExogenousStochasticBenchmark`](@ref).

This is the primary evaluation hook for stochastic benchmarks. The 2-arg fallback
`objective_value(bench, sample, y)` dispatches here using the scenario stored in
`sample.extra.scenario` (or averages over `sample.extra.scenarios`).
"""
function objective_value end

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
- Without `target_policy`: returns M×K unlabeled samples (`contexts_per_instance` contexts ×
  `nb_scenarios` scenarios each), each with one scenario in `extra=(; scenario=ξ)`.
- With `target_policy`: calls `target_policy(ctx_sample, scenarios)`
  per context and returns the result.

`target_policy(ctx_sample, scenarios) -> Vector{DataSample}` enables
anticipative labeling (K samples, one per scenario) or SAA (1 sample aggregating all K
scenarios).

!!! note
    This is an internal override of [`generate_sample`](@ref) for the stochastic pipeline,
    called by [`generate_dataset`](@ref). New stochastic benchmarks should implement
    [`generate_instance`](@ref), [`generate_context`](@ref), and [`generate_scenario`](@ref)
    rather than overriding this method. Note that the return type is `Vector{DataSample}`
    (one per context × scenario combination), unlike the base method which returns a
    single `DataSample`.
"""
function generate_sample(
    bench::ExogenousStochasticBenchmark,
    rng;
    target_policy=nothing,
    nb_scenarios::Int=1,
    contexts_per_instance::Int=1,
    kwargs...,
)
    instance_sample = generate_instance(bench, rng; kwargs...)
    return reduce(
        vcat,
        (
            let ctx = generate_context(bench, rng, instance_sample)
                if isnothing(target_policy)
                    [
                        DataSample(;
                            x=ctx.x,
                            θ=ctx.θ,
                            ctx.context...,
                            extra=(;
                                ctx.extra...,
                                scenario=generate_scenario(bench, rng; ctx.context...),
                            ),
                        ) for _ in 1:nb_scenarios
                    ]
                else
                    scenarios = [
                        generate_scenario(bench, rng; ctx.context...) for
                        _ in 1:nb_scenarios
                    ]
                    target_policy(ctx, scenarios)
                end
            end for _ in 1:contexts_per_instance
        ),
    )
end

"""
$TYPEDSIGNATURES

Specialised [`generate_dataset`](@ref) for exogenous stochastic benchmarks.

Generates `nb_instances` problem instances, each with `contexts_per_instance` context draws
and `nb_scenarios` scenario draws per context. The scenario→sample mapping is controlled
by the `target_policy`:
- Without `target_policy` (default): M contexts × K scenarios produce M×K unlabeled
  samples per instance.
- With `target_policy(ctx_sample, scenarios) -> Vector{DataSample}`:
  enables anticipative labeling (K labeled samples) or SAA (1 sample aggregating all K
  scenarios).

# Keyword arguments
- `nb_scenarios::Int = 1`: scenarios per context (K).
- `contexts_per_instance::Int = 1`: context draws per instance (M).
- `target_policy`: when provided, called as
  `target_policy(ctx_sample, scenarios)` to compute labels.
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
    contexts_per_instance::Int=1,
    seed=nothing,
    rng=MersenneTwister(seed),
    kwargs...,
)
    nb_instances == 0 && return DataSample[]
    return reduce(
        vcat,
        (
            generate_sample(
                bench, rng; target_policy, nb_scenarios, contexts_per_instance, kwargs...
            ) for _ in 1:nb_instances
        ),
    )
end

"""
$TYPEDEF

Transforms an [`ExogenousStochasticBenchmark`](@ref) into a static benchmark via
Sample Average Approximation (SAA).

For each (instance, context) pair, draws `nb_scenarios` fixed scenarios. These are stored
in the sample and used for feature computation, target labeling (via `target_policy`),
and gap evaluation.

!!! note
    `SampleAverageApproximation <: AbstractStaticBenchmark`, not `AbstractStochasticBenchmark`.
    This is intentional: after wrapping, the scenarios are fixed at dataset-generation time
    and the benchmark behaves as a static problem. Functions dispatching on
    `AbstractStochasticBenchmark` (e.g. `is_exogenous`) will not match SAA instances.

# Fields
$TYPEDFIELDS
"""
struct SampleAverageApproximation{B<:ExogenousStochasticBenchmark} <:
       AbstractStaticBenchmark
    "inner stochastic benchmark"
    benchmark::B
    "number of scenarios to draw per (instance, context) pair"
    nb_scenarios::Int
end

function is_minimization_problem(saa::SampleAverageApproximation)
    return is_minimization_problem(saa.benchmark)
end

function generate_maximizer(saa::SampleAverageApproximation; kwargs...)
    return generate_maximizer(saa.benchmark; kwargs...)
end

function generate_statistical_model(saa::SampleAverageApproximation; kwargs...)
    return generate_statistical_model(saa.benchmark; kwargs...)
end

function generate_sample(
    saa::SampleAverageApproximation, rng; target_policy=nothing, kwargs...
)
    inner = saa.benchmark
    instance_sample = generate_instance(inner, rng; kwargs...)
    ctx = generate_context(inner, rng, instance_sample)
    scenarios = [
        generate_scenario(inner, rng; ctx.context...) for _ in 1:(saa.nb_scenarios)
    ]
    if isnothing(target_policy)
        return [
            DataSample(;
                x=ctx.x, θ=ctx.θ, ctx.context..., extra=(; ctx.extra..., scenarios)
            ),
        ]
    else
        return target_policy(ctx, scenarios)
    end
end

"""
$TYPEDSIGNATURES

Specialised [`generate_dataset`](@ref) for [`SampleAverageApproximation`](@ref).

- Without `target_policy`: returns one static [`DataSample`](@ref) per instance, with
  `nb_scenarios` stored in `extra.scenarios`.
- With `target_policy(ctx_sample, scenarios) -> Vector{DataSample}`:
  labels each instance using all stored scenarios (same signature as
  [`ExogenousStochasticBenchmark`](@ref) policies).
"""
function generate_dataset(
    saa::SampleAverageApproximation,
    nb_instances::Int;
    target_policy=nothing,
    seed=nothing,
    rng=MersenneTwister(seed),
    kwargs...,
)
    nb_instances == 0 && return DataSample[]
    return reduce(
        vcat, (generate_sample(saa, rng; target_policy, kwargs...) for _ in 1:nb_instances)
    )
end

"""
$TYPEDSIGNATURES

Evaluate a decision `y` against stored scenarios (average over scenarios).
"""
function objective_value(
    saa::SampleAverageApproximation, sample::DataSample, y::AbstractArray
)
    return mean(
        objective_value(saa.benchmark, sample, y, ξ) for ξ in sample.extra.scenarios
    )
end

"""
$TYPEDSIGNATURES

Evaluate the target solution in the sample against stored scenarios.
"""
function objective_value(
    saa::SampleAverageApproximation, sample::DataSample{CTX,EX,F,S,C}
) where {CTX,EX,F,S<:AbstractArray,C}
    return objective_value(saa, sample, sample.y)
end
