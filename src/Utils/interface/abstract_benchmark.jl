"""
$TYPEDEF

Abstract root type for all benchmark problems.
"""
abstract type AbstractBenchmark end

"""
    generate_instance(::AbstractBenchmark, rng::AbstractRNG; kwargs...) -> DataSample

Generate a single unlabeled [`DataSample`](@ref) (with `y=nothing`) for the benchmark.
"""
function generate_instance(bench::AbstractBenchmark, rng::AbstractRNG; kwargs...)
    return error(
        "`generate_instance` is not implemented for $(typeof(bench)). " *
        "Implement `generate_instance(::$(typeof(bench)), rng; kwargs...) -> DataSample`. " *
        "For static benchmarks, you may also override `generate_sample` directly instead.",
    )
end

"""
    generate_maximizer(::AbstractBenchmark; kwargs...)

Returns a callable `f(θ; kwargs...) -> y`, solving a maximization problem.
"""
function generate_maximizer(bench::AbstractBenchmark; kwargs...)
    return error(
        "`generate_maximizer` is not implemented for $(typeof(bench)). " *
        "Implement `generate_maximizer(::$(typeof(bench)); kwargs...) -> f(θ; kwargs...) -> y`.",
    )
end

"""
    generate_statistical_model(::AbstractBenchmark, seed=nothing; kwargs...)

Returns an untrained statistical model (usually a Flux neural network) that maps a
feature matrix `x` to an output array `θ`. The `seed` parameter controls initialization
randomness for reproducibility.
"""
function generate_statistical_model(bench::AbstractBenchmark, seed=nothing; kwargs...)
    return error(
        "`generate_statistical_model` is not implemented for $(typeof(bench)). " *
        "Implement `generate_statistical_model(::$(typeof(bench)), seed=nothing; kwargs...) -> model`.",
    )
end

"""
    generate_baseline_policies(::AbstractBenchmark) -> NamedTuple or Tuple

Return named baseline policies for the benchmark. Each policy is a callable.
The calling convention matches the `target_policy` signature for the benchmark category:

- **Static:** `(sample) -> DataSample`
- **Stochastic:** `(ctx_sample, scenarios) -> Vector{DataSample}`
- **Dynamic:** `(env) -> Vector{DataSample}` (full trajectory rollout)
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
    objective_value(bench::AbstractBenchmark, sample::DataSample, y) -> Real

Compute the objective value of solution `y` for the benchmark instance encoded in `sample`.
Must be implemented by each concrete [`AbstractStaticBenchmark`](@ref).

For stochastic benchmarks, implement the 4-arg form instead (see
[`ExogenousStochasticBenchmark`](@ref)):

    objective_value(bench, sample, y, scenario) -> Real
"""
function objective_value end

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

Defaults to `true`. **Maximization benchmarks must override this method**, forgetting to do
so will cause `compute_gap` to compute the gap with the wrong sign without any error or warning.
"""
function is_minimization_problem(::AbstractBenchmark)
    return true
end

"""
$TYPEDSIGNATURES

Default implementation of [`compute_gap`](@ref): average relative optimality gap over `dataset`.
Requires labeled samples (`y ≠ nothing`), `x`, and `context` fields.
Override for custom evaluation logic.
"""
function compute_gap(
    bench::AbstractBenchmark,
    dataset::AbstractVector{<:DataSample{<:Any,<:Any,<:Any,<:AbstractArray}},
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
