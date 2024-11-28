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

struct StochasticVehicleSchedulingBenchmark <: AbstractBenchmark end

export StochasticVehicleSchedulingBenchmark
export create_random_city, compute_features

end
