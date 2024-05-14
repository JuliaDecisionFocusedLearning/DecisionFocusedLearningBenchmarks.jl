"""
$TYPEDEF

Abstract type for a benchmark problem.

The following methods need to be implemented:
- [`generate_dataset`](@ref)
- [`generate_statistical_model`](@ref)
- [`generate_maximizer`](@ref)
"""
abstract type AbstractBenchmark end

function generate_dataset end
function generate_statistical_model end
function generate_maximizer end
