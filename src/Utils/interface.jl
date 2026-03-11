"""
$TYPEDEF

Abstract type interface for benchmark problems.

# Mandatory methods to implement for any benchmark:
Choose one of three primary implementation strategies:
- Implement [`generate_instance`](@ref) (returns a [`DataSample`](@ref) with `y=nothing`).
  The default [`generate_sample`](@ref) then applies `target_policy` if provided.
- Override [`generate_sample`](@ref) directly when the sample requires custom logic. In this case,
  [`generate_dataset`](@ref) applies `target_policy` to the result after the call returns.
- Override [`generate_dataset`](@ref) directly when samples cannot be drawn independently.

Also implement:
- [`generate_statistical_model`](@ref)
- [`generate_maximizer`](@ref)

# Optional methods (defaults provided)
- [`is_minimization_problem`](@ref): defaults to `true`
- [`objective_value`](@ref): defaults to `dot(θ, y)`
- [`compute_gap`](@ref): default implementation provided; override for custom evaluation

# Optional methods (no default)
- [`plot_data`](@ref), [`plot_instance`](@ref), [`plot_solution`](@ref)
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
    generate_sample(::AbstractBenchmark, rng::AbstractRNG; target_policy=nothing, kwargs...) -> DataSample

Generate a single [`DataSample`](@ref) for the benchmark.

**Framework default** (when [`generate_instance`](@ref) is implemented):
Calls [`generate_instance`](@ref), then applies `target_policy(sample)` if provided.

Override directly (instead of implementing [`generate_instance`](@ref)) when the sample
requires custom logic. In this case, [`generate_dataset`](@ref) applies `target_policy`
after the call returns.
"""
function generate_sample(bench::AbstractBenchmark, rng; target_policy=nothing, kwargs...)
    sample = generate_instance(bench, rng; kwargs...)
    return isnothing(target_policy) ? sample : target_policy(sample)
end

"""
    generate_dataset(::AbstractBenchmark, dataset_size::Int; target_policy=nothing, kwargs...) -> Vector{<:DataSample}

Generate a `Vector` of [`DataSample`](@ref) of length `dataset_size` for given benchmark.
Content of the dataset can be visualized using [`plot_data`](@ref), when it applies.

By default, it uses [`generate_sample`](@ref) to create each sample in the dataset, and passes any
keyword arguments to it. If `target_policy` is provided, it is applied to each sample after
[`generate_sample`](@ref) returns.
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
- [`generate_instance`](@ref)`(bench, rng)`: returns a [`DataSample`](@ref) with instance
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

# Dataset generation (exogenous only)
[`generate_dataset`](@ref) is specialised for `AbstractStochasticBenchmark{true}` and
supports all three standard structures via `nb_scenarios`:

| Setting | Call |
|---------|------|
| 1 instance with K scenarios  | `generate_dataset(bench, 1; nb_scenarios=K)` |
| N instances with 1 scenario  | `generate_dataset(bench, N)` (default) |
| N instances with K scenarios | `generate_dataset(bench, N; nb_scenarios=K)` |

By default (no `target_policy`), each [`DataSample`](@ref) has `context` holding the
instance (solver kwargs) and `extra=(; scenario)` holding one scenario.

Provide a `target_policy(sample, scenarios) -> Vector{DataSample}` to compute labels.
This covers both anticipative (K samples, one per scenario) and SAA (1 sample from all K
scenarios) labeling strategies.
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
    generate_anticipative_solver(::AbstractStochasticBenchmark{true}) -> callable

Return a callable that computes the anticipative solution for a given scenario.
The instance and other solver-relevant fields are spread from the sample context.

- For [`AbstractStochasticBenchmark`](@ref): returns `(scenario; context...) -> y`.
- For [`AbstractDynamicBenchmark`](@ref): returns
  `(scenario; context...) -> Vector{DataSample}` — a full training trajectory.

    solver = generate_anticipative_solver(bench)
    y          = solver(scenario; sample.context...)  # stochastic
    trajectory = solver(scenario; sample.context...)  # dynamic
"""
function generate_anticipative_solver end

"""
    generate_parametric_anticipative_solver(::AbstractStochasticBenchmark{true}) -> callable

**Optional.** Return a callable `(θ, scenario; kwargs...) -> y` that solves the
parametric anticipative subproblem:

    argmin_{y ∈ Y(instance)}  c(y, scenario) + θᵀy
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

Default [`generate_sample`](@ref) for exogenous stochastic benchmarks.

Calls [`generate_instance`](@ref), draws `nb_scenarios` scenarios via
[`generate_scenario`](@ref), then:
- Without `target_policy`: returns K unlabeled samples, each with one scenario in
  `extra=(; scenario=ξ)`.
- With `target_policy`: calls `target_policy(sample, scenarios)` and returns the result.

`target_policy(sample, scenarios) -> Vector{DataSample}` enables anticipative labeling
(K samples, one per scenario) or SAA (1 sample aggregating all K scenarios).
"""
function generate_sample(
    bench::AbstractStochasticBenchmark{true},
    rng;
    target_policy=nothing,
    nb_scenarios::Int=1,
    kwargs...,
)
    sample = generate_instance(bench, rng; kwargs...)
    scenarios = [generate_scenario(bench, rng; sample.context...) for _ in 1:nb_scenarios]
    if isnothing(target_policy)
        return [
            DataSample(; x=sample.x, θ=sample.θ, sample.context..., extra=(; scenario=ξ))
            for ξ in scenarios
        ]
    else
        return target_policy(sample, scenarios)
    end
end

"""
$TYPEDSIGNATURES

Specialised [`generate_dataset`](@ref) for exogenous stochastic benchmarks.

Generates `nb_instances` problem instances, each with `nb_scenarios` independent
scenario draws. The scenario→sample mapping is controlled by the `target_policy`:
- Without `target_policy` (default): K scenarios produce K unlabeled samples (1:1).
- With `target_policy(sample, scenarios) -> Vector{DataSample}`: enables anticipative
  labeling (K labeled samples) or SAA (1 sample aggregating all K scenarios).

# Keyword arguments
- `nb_scenarios::Int = 1`: scenarios per instance (K).
- `target_policy`: when provided, called as `target_policy(sample, scenarios)` to
  compute labels. Defaults to `nothing` (unlabeled samples).
- `seed`: passed to `MersenneTwister` when `rng` is not provided.
- `rng`: random number generator; overrides `seed` when provided.
- `kwargs...`: forwarded to [`generate_sample`](@ref).
"""
function generate_dataset(
    bench::AbstractStochasticBenchmark{true},
    nb_instances::Int;
    target_policy=nothing,
    nb_scenarios::Int=1,
    seed=nothing,
    rng=MersenneTwister(seed),
    kwargs...,
)
    Random.seed!(rng, seed)
    samples = DataSample[]
    for _ in 1:nb_instances
        new_samples = generate_sample(bench, rng; target_policy, nb_scenarios, kwargs...)
        append!(samples, new_samples)
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
- [`generate_baseline_policies`](@ref)`(bench)`: returns named baseline callables of
  signature `(env) -> Vector{DataSample}` (full trajectory rollout).
- [`generate_dataset`](@ref)`(bench, environments; target_policy, ...)`: generates
  training-ready [`DataSample`](@ref)s by calling `target_policy(env)` for each environment.
  Requires `target_policy` as a mandatory keyword argument.
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
    bench::AbstractDynamicBenchmark{true},
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
    bench::AbstractDynamicBenchmark{true}, n::Int; target_policy, seed=nothing, kwargs...
)
    environments = generate_environments(bench, n; seed)
    return generate_dataset(bench, environments; target_policy, seed, kwargs...)
end
