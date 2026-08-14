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
    generate_statistical_model(::AbstractBenchmark; kwargs...)

Returns a Lux model architecture (no parameters) that maps features `x` to
an output array `θ`. Call `Lux.setup(rng, model)` to initialize parameters.
"""
function generate_statistical_model(bench::AbstractBenchmark; kwargs...)
    return error(
        "`generate_statistical_model` is not implemented for $(typeof(bench)). " *
        "Implement `generate_statistical_model(::$(typeof(bench)); kwargs...) -> model`.",
    )
end

"""
    generate_statistical_model(::AbstractBenchmark, rng::AbstractRNG; kwargs...)

Convenience method that returns `(model, ps, st)` ready to use.
Calls the base `generate_statistical_model(bench)` to get the architecture,
then initializes parameters with `Lux.setup(rng, model)`.
"""
function generate_statistical_model(bench::AbstractBenchmark, rng::AbstractRNG; kwargs...)
    model = generate_statistical_model(bench; kwargs...)
    ps, st = Lux.setup(rng, model)
    return model, ps, st
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

Return `true` if `plot_context` and `plot_sample` are implemented for this benchmark
(requires `Plots` to be loaded). Default is `false`.
"""
has_visualization(::AbstractBenchmark) = false

"""
    plot_context(bench::AbstractBenchmark, sample::DataSample; kwargs...)

Plot the observable context before making a decision (no solution). Only available when `Plots` is loaded.
"""
function plot_context end

"""
    plot_sample(bench::AbstractBenchmark, sample::DataSample; kwargs...)

Plot the instance with `sample.y` overlaid. Only available when `Plots` is loaded.
"""
function plot_sample end

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

"""
$TYPEDSIGNATURES

Convenience method accepting a Lux model with its parameters and state directly.
Switches to test mode and computes the gap inline.
"""
function compute_gap(
    bench::AbstractBenchmark,
    dataset::AbstractVector{<:DataSample{<:Any,<:Any,<:Any,<:AbstractArray}},
    model::Lux.AbstractLuxLayer,
    ps,
    st,
    maximizer,
    op=mean,
)
    check = is_minimization_problem(bench)
    st_test = Lux.testmode(st)

    return op(
        map(dataset) do sample
            target_obj = objective_value(bench, sample)
            θ = first(model(sample.x, ps, st_test))
            y = maximizer(θ; sample.context...)
            obj = objective_value(bench, sample, y)
            Δ = check ? obj - target_obj : target_obj - obj
            return Δ / abs(target_obj)
        end,
    )
end
