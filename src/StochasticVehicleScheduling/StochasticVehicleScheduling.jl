module StochasticVehicleScheduling

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Distributions: Distribution, LogNormal, Uniform
using Graphs: AbstractGraph, SimpleDiGraph, add_edge!, nv, ne, edges, src, dst
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
"""
function Utils.generate_dataset(
    benchmark::StochasticVehicleSchedulingBenchmark,
    dataset_size::Int;
    seed=nothing,
    rng=MersenneTwister(0),
)
    (; nb_tasks, nb_scenarios) = benchmark
    Random.seed!(rng, seed)
    instances = [Instance(; nb_tasks, nb_scenarios, rng=rng) for _ in 1:dataset_size]
    features = get_features.(instances)
    return [
        DataSample(; x=feature, instance) for
        (instance, feature) in zip(instances, features)
    ]
end

function Utils.generate_maximizer(bench::StochasticVehicleSchedulingBenchmark) end
function Utils.generate_statistical_model(bench::StochasticVehicleSchedulingBenchmark) end

export StochasticVehicleSchedulingBenchmark

export create_random_city, compute_features, Instance

end
