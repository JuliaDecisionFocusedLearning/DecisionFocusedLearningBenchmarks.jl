"""
$TYPEDEF

Abstract type interface for a benchmark problem.

The following methods exist for benchmarks:
- [`generate_dataset`](@ref)
- [`generate_statistical_model`](@ref)
- [`generate_maximizer`](@ref)
- [`plot_data`](@ref)
- [`objective_value`](@ref)
- [`compute_gap`](@ref)
"""
abstract type AbstractBenchmark end

"""
    generate_dataset(::AbstractBenchmark, dataset_size::Int) -> Vector{<:DataSample}

Generate a `Vector` of [`DataSample`](@ref)  of length `dataset_size` for given benchmark.
Content of the dataset can be visualized using [`plot_data`](@ref), when it applies.
"""
function generate_dataset end

"""
    generate_maximizer(::AbstractBenchmark)

Generates a maximizer function.
Returns a callable f: (θ; kwargs...) -> y, where θ is a cost array and y is a solution.
"""
function generate_maximizer end

"""
    generate_statistical_model(::AbstractBenchmark)

Initializes and return an untrained statistical model of the CO-ML pipeline.
It's usually a Flux model, that takes a feature matrix x as iinput, and returns a cost array θ as output.
"""
function generate_statistical_model end

"""
    plot_data(::AbstractBenchmark, args...)

Plot a data sample from the dataset created by [`generate_dataset`](@ref).
Check the specific benchmark documentation of `plot_data` for more details on the arguments.
"""
function plot_data end

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
    ::AbstractBenchmark, sample::DataSample{F,S,C,Nothing}
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

Compute the objective value of the target in the sample (needs to exist).
"""
function objective_value(
    bench::AbstractBenchmark, sample::DataSample{F,S,C,I}
) where {F,S<:AbstractArray,C<:AbstractArray,I}
    return objective_value(bench, sample.θ_true, sample.y_true)
end

"""
$TYPEDSIGNATURES

Compute the objective value of given solution `y`.
"""
function objective_value(
    bench::AbstractBenchmark, sample::DataSample{F,S,C,I}, y::AbstractArray
) where {F,S,C<:AbstractArray,I}
    return objective_value(bench, sample.θ_true, y)
end

"""
$TYPEDSIGNATURES

Default behaviour of `compute_gap` for a benchmark problem where `features`, `solutions` and `costs` are all defined.
"""
function compute_gap(
    bench::AbstractBenchmark, dataset::Vector{<:DataSample}, statistical_model, maximizer
)
    res = 0.0

    for sample in dataset
        target_obj = objective_value(bench, sample)
        x = sample.x
        θ = statistical_model(x)
        y = maximizer(θ; maximizer_kwargs(bench, sample)...)
        obj = objective_value(bench, sample, y)
        res += (target_obj - obj) / abs(target_obj)
    end
    return res / length(dataset)
end
