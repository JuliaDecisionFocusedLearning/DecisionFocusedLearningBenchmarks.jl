"""
$TYPEDEF

Abstract type for a benchmark problem.

The following methods should be implemented by most benchmarks:
- [`generate_dataset`](@ref)
- [`generate_statistical_model`](@ref)
- [`generate_maximizer`](@ref)
"""
abstract type AbstractBenchmark end

"""
Generate a dataset
"""
function generate_dataset end

"""
Returns the CO layer of the pipeline.
"""
function generate_maximizer end

"""
Initializes and return a statistical model of the pipeline.
"""
function generate_statistical_model end

"""
Plot a data sample from the dataset created by [`generate_dataset`](@ref).
"""
function plot_data end
