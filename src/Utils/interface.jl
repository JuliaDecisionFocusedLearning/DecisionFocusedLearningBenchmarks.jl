"""
$TYPEDEF

Abstract type interface for benchmark problems.

# Mandatory methods to implement for any benchmark:
- [`generate_sample`](@ref): primary entry point, called by the default [`generate_dataset`](@ref)
- [`generate_statistical_model`](@ref)
- [`generate_maximizer`](@ref)

Override [`generate_dataset`](@ref) directly only when samples cannot be drawn independently.

# Optional methods (defaults provided)
- [`is_minimization_problem`](@ref): defaults to `true`
- [`objective_value`](@ref): defaults to `dot(θ, y)`
- [`compute_gap`](@ref): default implementation provided; override for custom evaluation

# Optional methods (no default)
- [`plot_data`](@ref), [`plot_instance`](@ref), [`plot_solution`](@ref)
- [`generate_policies`](@ref)
"""
abstract type AbstractBenchmark end

"""
    generate_sample(::AbstractBenchmark, rng::AbstractRNG; kwargs...) -> DataSample

Generate a single [`DataSample`](@ref) for the benchmark.
This is the primary implementation target: the default [`generate_dataset`](@ref) calls
it repeatedly. Override [`generate_dataset`](@ref) directly only when samples cannot be
drawn independently (e.g. when the full dataset must be loaded at once).
"""
function generate_sample end

"""
    generate_dataset(::AbstractBenchmark, dataset_size::Int; kwargs...) -> Vector{<:DataSample}

Generate a `Vector` of [`DataSample`](@ref) of length `dataset_size` for given benchmark.
Content of the dataset can be visualized using [`plot_data`](@ref), when it applies.

By default, it uses [`generate_sample`](@ref) to create each sample in the dataset, and passes any
keyword arguments to it.
"""
function generate_dataset(
    bench::AbstractBenchmark,
    dataset_size::Int;
    seed=nothing,
    rng=MersenneTwister(seed),
    kwargs...,
)
    Random.seed!(rng, seed)
    return [generate_sample(bench, rng; kwargs...) for _ in 1:dataset_size]
end

"""
    generate_maximizer(::AbstractBenchmark; kwargs...)

Returns a callable `f(θ; kwargs...) -> y`, solving a maximization problem.
"""
function generate_maximizer end

"""
    generate_statistical_model(::AbstractBenchmark; kwargs...)

Returns an untrained statistical model (usually a Flux neural network) that maps a
feature matrix `x` to an output array `θ`.
"""
function generate_statistical_model end

"""
    generate_policies(::AbstractBenchmark) -> Vector{Policy}

Return a list of named baseline policies for the benchmark.
"""
function generate_policies end

"""
    plot_data(::AbstractBenchmark, ::DataSample; kwargs...)

Plot a data sample from the dataset created by [`generate_dataset`](@ref).
Check the specific benchmark documentation of `plot_data` for more details on the arguments.
"""
function plot_data end

"""
    plot_instance(::AbstractBenchmark, instance; kwargs...)

Plot the instance object of the sample.
"""
function plot_instance end

"""
    plot_solution(::AbstractBenchmark, sample::DataSample, [solution]; kwargs...)

Plot `solution` if given, else plot the target solution in the sample.
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
            y = maximizer(θ; sample.context...)
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

# Required methods (exogenous benchmarks, `{true}` only)
- [`generate_sample`](@ref)`(bench, rng)`: returns a [`DataSample`](@ref) with instance
  and features but **no scenario**. Scenarios are added later by [`generate_dataset`](@ref)
  via [`generate_scenario`](@ref).
- [`generate_scenario`](@ref)`(bench, rng; kwargs...)`: draws a random scenario.
  Instance and context fields are passed as keyword arguments spread from `sample.context`.

# Optional methods
- [`generate_anticipative_solver`](@ref)`(bench)`: returns a callable
  `(scenario; kwargs...) -> y` that computes the anticipative solution per scenario.
- [`generate_parametric_anticipative_solver`](@ref)`(bench)`: returns a callable
  `(θ, scenario; kwargs...) -> y` for the parametric anticipative subproblem
  `argmin_{y ∈ Y} c(y, scenario) + θᵀy`.
- [`generate_instance_samples`](@ref)`(bench, sample, scenarios; compute_targets,
  kwargs...)`: maps K scenarios to `DataSample`s for one instance. Override to change
  the scenario→sample mapping (e.g. SAA: K scenarios → 1 sample with shared target).

# Dataset generation (exogenous only)
[`generate_dataset`](@ref) is specialised for `AbstractStochasticBenchmark{true}` and
supports all three standard structures via `nb_scenarios_per_instance`:

| Setting | Call |
|---------|------|
| 1 instance with K scenarios  | `generate_dataset(bench, 1; nb_scenarios_per_instance=K)` |
| N instances with 1 scenario  | `generate_dataset(bench, N)` (default) |
| N instances with K scenarios | `generate_dataset(bench, N; nb_scenarios_per_instance=K)` |

By default, each [`DataSample`](@ref) has `context` holding the instance (solver kwargs)
and `extra=(; scenario)` holding one scenario.  Override
[`generate_instance_samples`](@ref) to store scenarios differently.
"""
abstract type AbstractStochasticBenchmark{exogenous} <: AbstractBenchmark end

is_exogenous(::AbstractStochasticBenchmark{exogenous}) where {exogenous} = exogenous
is_endogenous(::AbstractStochasticBenchmark{exogenous}) where {exogenous} = !exogenous

"""
    generate_scenario(::AbstractStochasticBenchmark{true}, rng::AbstractRNG; kwargs...) -> scenario

Draw a random scenario. Instance and context fields are passed as keyword arguments,
spread from `sample.context`:

    scenario = generate_scenario(bench, rng; sample.context...)
"""
function generate_scenario end

"""
    generate_anticipative_solver(::AbstractStochasticBenchmark) -> callable

Return a callable that computes the anticipative solution for a given scenario.
The instance and other solver-relevant fields are spread from the sample context.

- For [`AbstractStochasticBenchmark`](@ref): returns `(scenario; kwargs...) -> y`.
- For [`AbstractDynamicBenchmark`](@ref): returns
  `(scenario; kwargs...) -> Vector{DataSample}` — a full training trajectory.

    solver = generate_anticipative_solver(bench)
    y          = solver(scenario; sample.context...)  # stochastic
    trajectory = solver(scenario; sample.context...)  # dynamic
"""
function generate_anticipative_solver(bench::AbstractStochasticBenchmark)
    return (scenario; kwargs...) -> error(
        "`generate_anticipative_solver` is not implemented for $(typeof(bench)). " *
        "Implement `generate_anticipative_solver(::$(typeof(bench))) -> (scenario; kwargs...) -> y` " *
        "to use `compute_targets=true`.",
    )
end

"""
    generate_parametric_anticipative_solver(::AbstractStochasticBenchmark) -> callable

**Optional.** Return a callable `(θ, scenario; kwargs...) -> y` that solves the
parametric anticipative subproblem:

    argmin_{y ∈ Y(instance)}  c(y, scenario) + θᵀy

The scenario comes first (it defines the stochastic cost function); `θ` is the
perturbation added on top, coupling the benchmark to the model output.
"""
function generate_parametric_anticipative_solver end

"""
    generate_anticipative_solution(::AbstractStochasticBenchmark, instance, scenario; kwargs...)

!!! warning "Deprecated"
    Use [`generate_anticipative_solver`](@ref) instead, which returns a callable
    `(scenario; kwargs...) -> y` consistent with the [`generate_maximizer`](@ref)
    convention.
"""
function generate_anticipative_solution end

"""
$TYPEDSIGNATURES

Map K scenarios to [`DataSample`](@ref)s for a single instance (encoded in `sample`).

This is the key customisation point for scenario→sample mapping in
[`generate_dataset`](@ref).

**Default** (1:1 mapping):
Returns K samples, each with one scenario in `extra=(; scenario=ξ)`.
When `compute_targets=true`, calls [`generate_anticipative_solver`](@ref) to compute
an independent anticipative target per scenario.

**Override for batch strategies** (e.g. SAA):
Return fewer samples (or one) using all K scenarios together.  Extra keyword arguments
forwarded from [`generate_dataset`](@ref) reach here, enabling solver choice:

```julia
function generate_instance_samples(bench::MySAABench, sample, scenarios;
                                    compute_targets=false, algorithm=my_solver, kwargs...)
    y = compute_targets ? algorithm(sample.instance, scenarios; kwargs...) : nothing
    return [DataSample(; x=sample.x, θ=sample.θ, y, sample.context...,
                        extra=(; scenarios))]
end
```
"""
function generate_instance_samples(
    bench::AbstractStochasticBenchmark{true},
    sample::DataSample,
    scenarios::AbstractVector;
    compute_targets::Bool=false,
    kwargs...,
)
    solver = generate_anticipative_solver(bench)
    return [
        DataSample(;
            x=sample.x,
            θ=sample.θ,
            y=compute_targets ? solver(ξ; sample.context...) : nothing,
            sample.context...,
            extra=(; scenario=ξ),
        ) for ξ in scenarios
    ]
end

"""
$TYPEDSIGNATURES

Specialised [`generate_dataset`](@ref) for exogenous stochastic benchmarks.

Generates `nb_instances` problem instances, each with `nb_scenarios_per_instance`
independent scenario draws.  The scenario→sample mapping is controlled by
[`generate_instance_samples`](@ref): by default K scenarios produce K samples
(1:1, anticipative), but overriding it enables batch strategies such as SAA
(K scenarios → 1 sample with a shared target).

# Keyword arguments
- `nb_scenarios_per_instance::Int = 1` — scenarios per instance (K).
- `compute_targets::Bool = false` — when `true`, passed to
  [`generate_instance_samples`](@ref) to trigger target computation.
- `seed` — passed to `MersenneTwister` when `rng` is not provided.
- `rng` — random number generator; overrides `seed` when provided.
- `kwargs...` — forwarded to [`generate_instance_samples`](@ref) (e.g. `algorithm=...`).
"""
function generate_dataset(
    bench::AbstractStochasticBenchmark{true},
    nb_instances::Int;
    nb_scenarios_per_instance::Int=1,
    compute_targets::Bool=false,
    seed=nothing,
    rng=MersenneTwister(seed),
    kwargs...,
)
    Random.seed!(rng, seed)
    samples = DataSample[]
    for _ in 1:nb_instances
        sample = generate_sample(bench, rng)
        scenarios = [
            generate_scenario(bench, rng; sample.context...) for
            _ in 1:nb_scenarios_per_instance
        ]
        append!(
            samples,
            generate_instance_samples(bench, sample, scenarios; compute_targets, kwargs...),
        )
    end
    return samples
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
- [`generate_scenario`](@ref)`(bench, rng; kwargs...)`: for `{true}` (exogenous) benchmarks:
  draw a full multi-stage scenario. Instance is passed as `instance=env.instance` keyword.
  Required only when using [`generate_dataset`](@ref)`(bench, environments; ...)`.
- [`generate_anticipative_solver`](@ref)`(bench)`: returns a callable
  `(scenario; kwargs...) -> Vector{DataSample}` that runs the anticipative solver on a
  full scenario and returns a training trajectory. Required only when using
  [`generate_dataset`](@ref)`(bench, environments; ...)`.
- [`generate_dataset`](@ref)`(bench, environments; nb_scenarios_per_env, ...)`: optional;
  generates training-ready [`DataSample`](@ref)s from environments via anticipative rollouts.
  Requires [`generate_scenario`](@ref) and [`generate_anticipative_solver`](@ref).
"""
abstract type AbstractDynamicBenchmark{exogenous} <: AbstractStochasticBenchmark{exogenous} end

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

Map K scenarios to training [`DataSample`](@ref)s for a single environment.

Key customisation point for scenario→sample mapping in
[`generate_dataset`](@ref)`(bench, environments; nb_scenarios_per_env)`.

**Default:** Calls [`generate_anticipative_solver`](@ref) on each scenario,
returning the concatenated trajectories.

**Override for custom strategies** (e.g. averaging trajectories, sub-sampling steps).
"""
function generate_environment_samples(
    bench::AbstractDynamicBenchmark{true}, env, scenarios::AbstractVector; kwargs...
)
    solver = generate_anticipative_solver(bench)
    samples = DataSample[]
    for scenario in scenarios
        trajectory = solver(scenario; instance=env.instance, kwargs...)
        append!(samples, trajectory)
    end
    return samples
end

"""
$TYPEDSIGNATURES

Generate a training dataset from pre-built environments for an exogenous dynamic benchmark.

For each environment, draws `nb_scenarios_per_env` independent scenarios via
[`generate_scenario`](@ref) and maps them to [`DataSample`](@ref)s via
[`generate_environment_samples`](@ref).

Mirrors [`generate_dataset`](@ref) for [`AbstractStochasticBenchmark{true}`](@ref)
with `nb_scenarios_per_instance`.

| | Stochastic | Dynamic |
|---|---|---|
| Unit | instance (`DataSample`) | environment |
| Customisation hook | `generate_instance_samples` | `generate_environment_samples` |
"""
function generate_dataset(
    bench::AbstractDynamicBenchmark{true},
    environments::AbstractVector;
    nb_scenarios_per_env::Int=1,
    seed=nothing,
    rng=MersenneTwister(seed),
    kwargs...,
)
    Random.seed!(rng, seed)
    samples = DataSample[]
    for env in environments
        scenarios = [
            generate_scenario(bench, rng; instance=env.instance) for
            _ in 1:nb_scenarios_per_env
        ]
        append!(samples, generate_environment_samples(bench, env, scenarios; kwargs...))
    end
    return samples
end

"""
$TYPEDSIGNATURES

Convenience wrapper for exogenous dynamic benchmarks: generates `n` environments
via [`generate_environments`](@ref), then calls
[`generate_dataset`](@ref)`(bench, environments; nb_scenarios_per_env, ...)`.

Gives dynamic benchmarks the same top-level API as static/stochastic:

    dataset = generate_dataset(bench, n)
    dataset = generate_dataset(bench, n; nb_scenarios_per_env=5)
"""
function generate_dataset(
    bench::AbstractDynamicBenchmark{true},
    n::Int;
    nb_scenarios_per_env::Int=1,
    seed=nothing,
    kwargs...,
)
    environments = generate_environments(bench, n; seed)
    return generate_dataset(bench, environments; nb_scenarios_per_env, seed, kwargs...)
end
