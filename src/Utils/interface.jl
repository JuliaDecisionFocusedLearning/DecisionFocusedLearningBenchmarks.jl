"""
$TYPEDEF

Abstract type for a benchmark problem.

The following methods should be implemented by most benchmarks:
- [`generate_dataset`](@ref)
- [`generate_statistical_model`](@ref)
- [`generate_maximizer`](@ref)
"""
abstract type AbstractBenchmark end

function generate_dataset end
function generate_maximizer end
function generate_statistical_model end
