module StochasticVehicleScheduling

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using ConstrainedShortestPaths:
    stochastic_routing_shortest_path, stochastic_routing_shortest_path_with_threshold
using Distributions: Distribution, LogNormal, Uniform
using Flux: Chain, Dense
using Graphs:
    AbstractGraph,
    SimpleDiGraph,
    add_edge!,
    nv,
    ne,
    edges,
    src,
    dst,
    has_edge,
    inneighbors,
    outneighbors
using JuMP:
    Model,
    @variable,
    @objective,
    @constraint,
    optimize!,
    value,
    objective_value,
    set_silent,
    dual
using Printf: @printf
using Random: Random, AbstractRNG, MersenneTwister
using SparseArrays: sparse
using Statistics: quantile, mean

include("utils.jl")
include("instance/constants.jl")
include("instance/task.jl")
include("instance/district.jl")
include("instance/city.jl")
include("instance/features.jl")
include("instance/instance.jl")

include("solution/solution.jl")
include("solution/exact_algorithms/mip.jl")
include("solution/exact_algorithms/column_generation.jl")

include("maximizer.jl")

"""
$TYPEDFIELDS

Data structure for a stochastic vehicle scheduling benchmark.

# Fields
$TYPEDFIELDS
"""
@kwdef struct StochasticVehicleSchedulingBenchmark <: AbstractBenchmark
    "number of tasks in each instance"
    nb_tasks::Int = 25
    "number of scenarios in each instance"
    nb_scenarios::Int = 10
end

"""
$TYPEDSIGNATURES

Create a dataset of `dataset_size` instances for the given `StochasticVehicleSchedulingBenchmark`.
If you want to also add label solutions in the dataset, set `compute_solutions=true`.
By default, they will be computed using column generation.
Note that computing solutions can be time-consuming, especially for large instances.
You can also use instead `compact_mip` or `compact_linearized_mip` as the algorithm to compute solutions.
If you want to provide a custom algorithm to compute solutions, you can pass it as the `algorithm` keyword argument.
If `algorithm` takes keyword arguments, you can pass them as well directly in `kwargs...`.
"""
function Utils.generate_dataset(
    benchmark::StochasticVehicleSchedulingBenchmark,
    dataset_size::Int;
    compute_solutions=false,
    seed=nothing,
    rng=MersenneTwister(0),
    algorithm=column_generation_algorithm,
    kwargs...,
)
    (; nb_tasks, nb_scenarios) = benchmark
    Random.seed!(rng, seed)
    instances = [Instance(; nb_tasks, nb_scenarios, rng=rng) for _ in 1:dataset_size]
    features = get_features.(instances)
    if compute_solutions
        solutions = [algorithm(instance; kwargs...) for instance in instances]
        return [
            DataSample(; x=feature, instance, y=solution) for
            (instance, feature, solution) in zip(instances, features, solutions)
        ]
    end
    # else
    return [
        DataSample(; x=feature, instance) for
        (instance, feature) in zip(instances, features)
    ]
end

"""
$TYPEDSIGNATURES
"""
function Utils.generate_maximizer(bench::StochasticVehicleSchedulingBenchmark)
    return vsp_maximizer
end

"""
$TYPEDSIGNATURES
"""
function Utils.generate_statistical_model(bench::StochasticVehicleSchedulingBenchmark)
    return Chain(Dense(20 => 1; bias=false), vec)
end

export StochasticVehicleSchedulingBenchmark
export generate_dataset, generate_maximizer, generate_statistical_model
export compact_linearized_mip,
    compact_mip, column_generation_algorithm, evaluate_solution, is_feasible

end
