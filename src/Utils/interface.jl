"""
$TYPEDEF

Abstract type interface for a benchmark problem.

The following methods should be implemented by most benchmarks:
- [`generate_dataset`](@ref)
- [`generate_statistical_model`](@ref)
- [`generate_maximizer`](@ref)
- [`plot_data`](@ref)
"""
abstract type AbstractBenchmark end

"""
    generate_dataset(::AbstractBenchmark, dataset_size::Int) -> Vector{NamedTuple}

Generate a dataset for given benchmark as a Vector of length `dataset_size`.
Each element is a `NamedTuple` with possible fields:
- `features`: feature matrix (always present)
- `solution`: groundtruth solution (optional)
- `costs`: groundtruth cost array (optional)
- `instance`: instance of the problem (optional)

Check the specific benchmark documentation of `generate_dataset` for more details on the fields.

Each element of the dataset can be visualized using [`plot_data`](@ref).
"""
function generate_dataset end

"""
    generate_maximizer(::AbstractBenchmark)

Generates a maximizer function.
Returns a callable f: θ -> argmax θᵀy, where θ is a cost array and y is a solution.
"""
function generate_maximizer end

"""
    generate_statistical_model(::AbstractBenchmark)

Initializes and return an untrained statistical model of the CO-ML pipeline.
It's usually a flux model, that takes a feature matrix x as iinput, and returns a cost array θ as output.
"""
function generate_statistical_model end

"""
    plot_data(::AbstractBenchmark, args...)

Plot a data sample from the dataset created by [`generate_dataset`](@ref).
Check the specific benchmark documentation of `plot_data` for more details on the arguments.
"""
function plot_data end
