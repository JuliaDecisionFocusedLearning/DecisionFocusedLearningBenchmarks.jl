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
    rng=MersenneTwister(0),
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

For simple benchmarks where there is no instance object, maximizer does not need any keyword arguments.
"""
function maximizer_kwargs(
    ::AbstractBenchmark, sample::DataSample{Nothing,F,S,C}
) where {F,S,C}
    return NamedTuple()
end

"""
$TYPEDSIGNATURES

For benchmarks where there is an instance object, maximizer needs the instance object as a keyword argument.
"""
function maximizer_kwargs(::AbstractBenchmark, sample::DataSample)
    return (; instance=sample.instance)
end

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
    bench::AbstractBenchmark, sample::DataSample{I,F,S,C}, y::AbstractArray
) where {I,F,S,C<:AbstractArray}
    return objective_value(bench, sample.θ_true, y)
end

"""
$TYPEDSIGNATURES

Compute the objective value of the target in the sample (needs to exist).
"""
function objective_value(
    bench::AbstractBenchmark, sample::DataSample{I,F,S,C}
) where {I,F,S<:AbstractArray,C}
    return objective_value(bench, sample, sample.y_true)
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
            y = maximizer(θ; maximizer_kwargs(bench, sample)...)
            obj = objective_value(bench, sample, y)
            Δ = check ? obj - target_obj : target_obj - obj
            return Δ / abs(target_obj)
        end,
    )
end

"""
$TYPEDEF

Abstract type interface for stochastic benchmark problems.
This type should be used for benchmarks that involve single stage stochastic optimization problems.

It follows the same interface as [`AbstractBenchmark`](@ref), with the addition of the following methods:
- [`generate_anticipative_solver`](@ref)
"""
abstract type AbstractStochasticBenchmark <: AbstractBenchmark end

# only works for exogenous noise
"""
    generate_scenario(::AbstractStochasticBenchmark; kwargs...)
"""
function generate_scenario_generator end

"""
    generate_anticipative_solver(::AbstractStochasticBenchmark; kwargs...)
"""
function generate_anticipative_solver end

"""
$TYPEDEF

Abstract type interface for dynamic benchmark problems.
This type should be used for benchmarks that involve multi-stage stochastic optimization problems.

It follows the same interface as [`AbstractStochasticBenchmark`](@ref), with the addition of the following methods:
TODO
"""
abstract type AbstractDynamicBenchmark <: AbstractStochasticBenchmark end
