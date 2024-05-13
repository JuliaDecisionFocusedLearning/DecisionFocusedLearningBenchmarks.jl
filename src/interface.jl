"""
$TYPEDEF

Abstract benchmark type.

# Interface
Each subtype should have the following fields (or overwrite corresponding methods):
- `features`: vector of feature arrays
- `optimization_parameters`: label 'true' optimization parameters
- `solutions`: label solutions
- `maximizer`: (combinatorial) optimization (arg) maximizer
"""
abstract type AbstractBenchmark end

"""
$TYPEDSIGNATURES
"""
get_features(bench::AbstractBenchmark) = bench.features

"""
$TYPEDSIGNATURES
"""
get_optimization_parameters(bench::AbstractBenchmark) = bench.optimization_parameters

"""
$TYPEDSIGNATURES
"""
get_solutions(bench::AbstractBenchmark) = bench.solutions

"""
$TYPEDSIGNATURES
"""
get_maximizer(bench::AbstractBenchmark) = bench.maximizer

input_size(bench::AbstractBenchmark) = length(get_features(bench)[1])
output_size(bench::AbstractBenchmark) = length(get_solutions(bench)[1])
