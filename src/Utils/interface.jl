"""
$TYPEDEF

Abstract type interface for benchmark problems.

The following methods are mandatory for benchmarks:
- [`generate_dataset`](@ref) or [`generate_sample`](@ref)
- [`generate_statistical_model`](@ref)
- [`generate_maximizer`](@ref)

The following methods are optional:
- [`plot_data`](@ref)
- [`objective_value`](@ref)
- [`compute_gap`](@ref)
"""
abstract type AbstractBenchmark end

"""
    generate_sample(::AbstractBenchmark, rng::AbstractRNG; kwargs...) -> DataSample

Generate a single [`DataSample`](@ref) for given benchmark.
This is a low-level function that is used by [`generate_dataset`](@ref) to create
a dataset of samples. It is not mandatory to implement this method, but it is
recommended for benchmarks that have a well-defined way to generate individual samples.
An alternative is to directly implement [`generate_dataset`](@ref) to create a dataset
without generating individual samples.
"""
function generate_sample end

"""
    generate_dataset(::AbstractBenchmark, dataset_size::Int; kwargs...) -> Vector{<:DataSample}

Generate a `Vector` of [`DataSample`](@ref) of length `dataset_size` for given benchmark.
Content of the dataset can be visualized using [`plot_data`](@ref), when it applies.

By default, it uses [`generate_sample`](@ref) to create each sample in the dataset, and passes any keyword arguments to it.
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

Generates a maximizer function.
Returns a callable f: (θ; kwargs...) -> y, where θ is a cost array and y is a solution.
"""
function generate_maximizer end

"""
    generate_statistical_model(::AbstractBenchmark; kwargs...)

Initializes and return an untrained statistical model of the CO-ML pipeline.
It's usually a Flux model, that takes a feature matrix x as input, and returns a cost array θ as output.
"""
function generate_statistical_model end

"""
    generate_policies(::AbstractBenchmark) -> Vector{Policy}
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

Default behaviour of `objective_value`.
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

Default behaviour of `compute_gap` for a benchmark problem where `features`, `solutions` and `costs` are all defined.
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

A stochastic benchmark separates the problem into a **deterministic instance** (the
context known before the scenario is revealed) and a **random scenario** (the uncertain
part). The combinatorial oracle sees only the instance; scenarios are used to evaluate
anticipative solutions, generate targets, and compute objective values.

# Required methods (exogenous benchmarks, `{true}` only)
- [`generate_sample`](@ref)`(bench, rng)`: returns a [`DataSample`](@ref) with instance
  and features but **no scenario**.  The scenario is omitted so that
  [`generate_dataset`](@ref) can draw K independent scenarios from the same instance.
- [`generate_scenario`](@ref)`(bench, sample, rng)`: draws a random scenario for the
  instance encoded in `sample`.  The full sample is passed (not just the instance)
  because context is tied to the instance and implementations may need fields beyond
  `sample.instance`.

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

Extra keyword arguments are forwarded to [`generate_instance_samples`](@ref), enabling
solver choice to reach target computation (e.g. `algorithm=compact_mip`).

By default, each [`DataSample`](@ref) has `context` holding the instance (solver kwargs)
and `extra=(; scenario)` holding one scenario.  Override
[`generate_instance_samples`](@ref) to store scenarios differently (e.g.
`extra=(; scenarios=[ξ₁,…,ξ_K])` for SAA).
"""
abstract type AbstractStochasticBenchmark{exogenous} <: AbstractBenchmark end

is_exogenous(::AbstractStochasticBenchmark{exogenous}) where {exogenous} = exogenous
is_endogenous(::AbstractStochasticBenchmark{exogenous}) where {exogenous} = !exogenous

"""
    generate_scenario(::AbstractStochasticBenchmark{true}, sample::DataSample,
                      rng::AbstractRNG) -> scenario

Draw a random scenario for the instance encoded in `sample`.
Called once per scenario by the specialised [`generate_dataset`](@ref).

The full `sample` is passed (not just `sample.instance`) because both the scenario
and the context are tied to the same instance — implementations may need any field
of the sample.  Consistent with [`generate_environment`](@ref) for dynamic benchmarks.
"""
function generate_scenario end

"""
    generate_anticipative_solver(::AbstractStochasticBenchmark) -> callable

Return a callable `(scenario; kwargs...) -> y` that computes the anticipative solution for a given
scenario. The instance and other solver-relevant fields are spread from the sample context:

    solver = generate_anticipative_solver(bench)
    y = solver(scenario; sample.context...)

This mirrors the maximizer calling convention `maximizer(θ; sample.context...)`.

Used by Imitating Anticipative and DAgger algorithms.  Replaces the deprecated
[`generate_anticipative_solution`](@ref).
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

The κ weight from the Alternating Minimization algorithm is not a parameter of this
solver.  Since the subproblem is linear in `θ`, the algorithm scales θ by κ before
calling: `solver(κ * θ, scenario; sample.context...)`.

Partially apply `scenario` to obtain a `(θ; kwargs...) -> y` closure, then wrap in
`PerturbedAdditive` (InferOpt) to compute targets `μᵢ` during the decomposition step.
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

**Default** (anticipative / DAgger — 1:1 mapping):
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
            generate_scenario(bench, sample, rng) for _ in 1:nb_scenarios_per_instance
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

Abstract type interface for dynamic benchmark problems.
This type should be used for benchmarks that involve multi-stage stochastic optimization problems.

It follows the same interface as [`AbstractStochasticBenchmark`](@ref), with the addition of the following methods:
TODO
"""
abstract type AbstractDynamicBenchmark{exogenous} <: AbstractStochasticBenchmark{exogenous} end

# Dynamic benchmarks do not use the stochastic dataset generation (which draws independent
# scenarios per instance). They generate each sample independently via `generate_sample`,
# using the standard AbstractBenchmark default.
function generate_dataset(
    bench::AbstractDynamicBenchmark,
    dataset_size::Int;
    seed=nothing,
    rng=MersenneTwister(seed),
    kwargs...,
)
    Random.seed!(rng, seed)
    return [generate_sample(bench, rng; kwargs...) for _ in 1:dataset_size]
end

# Dynamic benchmarks generate complete trajectories via `generate_sample` and do not
# decompose problems into (instance, scenario) pairs. `generate_scenario` is not
# applicable to them; this method exists only to provide a clear error.
function generate_scenario(
    bench::AbstractDynamicBenchmark, sample::DataSample, rng::AbstractRNG; kwargs...
)
    return error(
        "`generate_scenario` is not applicable to dynamic benchmarks ($(typeof(bench))). " *
        "Dynamic benchmarks generate complete trajectories via `generate_sample`.",
    )
end

"""
    generate_environment(::AbstractDynamicBenchmark, instance, rng::AbstractRNG; kwargs...)

Initialize an environment for the given dynamic benchmark instance.
"""
function generate_environment end

"""
$TYPEDSIGNATURES

Default behaviour of `generate_environment` applied to a data sample.
Uses the info field of the sample as the instance.
"""
function generate_environment(
    bench::AbstractDynamicBenchmark, sample::DataSample, rng::AbstractRNG; kwargs...
)
    return generate_environment(bench, sample.instance, rng; kwargs...)
end

"""
$TYPEDSIGNATURES

Generate a vector of environments for the given dynamic benchmark and dataset.
"""
function generate_environments(
    bench::AbstractDynamicBenchmark,
    dataset::AbstractArray;
    seed=nothing,
    rng=MersenneTwister(seed),
    kwargs...,
)
    Random.seed!(rng, seed)
    return map(dataset) do sample
        generate_environment(bench, sample, rng; kwargs...)
    end
end
