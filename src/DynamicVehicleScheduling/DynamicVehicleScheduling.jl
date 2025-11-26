module DynamicVehicleScheduling

using ..Utils

using Base: @kwdef
using DataDeps: @datadep_str
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Flux: Chain, Dense
using Graphs
using HiGHS
using InferOpt: LinearMaximizer
using IterTools: partition
using JSON
using JuMP
using Plots: plot, plot!, scatter!, @animate, Plots, gif
using Printf: @printf, @sprintf
using Random: Random, AbstractRNG, MersenneTwister, seed!, randperm
using Requires: @require
using Statistics: mean, quantile

include("utils.jl")

# static vsp stuff
include("static_vsp/instance.jl")
include("static_vsp/parsing.jl")
include("static_vsp/solution.jl")
include("static_vsp/plot.jl")

include("instance.jl")
include("state.jl")
include("scenario.jl")
include("environment.jl")
include("plot.jl")

include("maximizer.jl")
include("anticipative_solver.jl")

include("features.jl")
include("policy.jl")

"""
$TYPEDEF

Abstract type for dynamic vehicle scheduling benchmarks.

# Fields
$TYPEDFIELDS
"""
@kwdef struct DynamicVehicleSchedulingBenchmark <: AbstractDynamicBenchmark{true}
    "maximum number of customers entering the system per epoch"
    max_requests_per_epoch::Int = 10
    "time between decision and dispatch of a vehicle"
    Δ_dispatch::Float64 = 1.0
    "duration of an epoch"
    epoch_duration::Float64 = 1.0
    "whether to use two-dimensional features"
    two_dimensional_features::Bool = false
end

"""
$TYPEDSIGNATURES

Generate a dataset for the dynamic vehicle scheduling benchmark.
Returns a vector of [`DataSample`](@ref) objects, each containing an [`Instance`](@ref).
The dataset is generated from pre-existing DVRPTW files.
"""
function Utils.generate_dataset(b::DynamicVehicleSchedulingBenchmark, dataset_size::Int=1)
    (; max_requests_per_epoch, Δ_dispatch, epoch_duration, two_dimensional_features) = b
    files = readdir(datadep"dvrptw"; join=true)
    dataset_size = min(dataset_size, length(files))
    return [
        DataSample(;
            instance=Instance(
                read_vsp_instance(files[i]);
                max_requests_per_epoch,
                Δ_dispatch,
                epoch_duration,
                two_dimensional_features,
            ),
        ) for i in 1:dataset_size
    ]
end

"""
$TYPEDSIGNATURES

Creates an environment from an [`Instance`](@ref) of the dynamic vehicle scheduling benchmark.
The seed of the environment is randomly generated using the provided random number generator.
"""
function Utils.generate_environment(
    ::DynamicVehicleSchedulingBenchmark, instance::Instance, rng::AbstractRNG; kwargs...
)
    seed = rand(rng, 1:typemax(Int))
    return DVSPEnv(instance; seed)
end

"""
$TYPEDSIGNATURES

Returns a linear maximizer for the dynamic vehicle scheduling benchmark, of the form:
θ ↦ argmax_{y} θᵀg(y) + h(y)
"""
function Utils.generate_maximizer(::DynamicVehicleSchedulingBenchmark)
    return LinearMaximizer(oracle; g, h)
end

"""
$TYPEDSIGNATURES

Generate a scenario for the dynamic vehicle scheduling benchmark.
This is a wrapper around the generic scenario generation function.
"""
function Utils.generate_scenario(b::DynamicVehicleSchedulingBenchmark, args...; kwargs...)
    return Utils.generate_scenario(args...; kwargs...)
end

"""
$TYPEDSIGNATURES

Generate an anticipative solution for the dynamic vehicle scheduling benchmark.
The solution is computed using the anticipative solver with the benchmark's feature configuration.
"""
function Utils.generate_anticipative_solution(
    b::DynamicVehicleSchedulingBenchmark, args...; kwargs...
)
    return anticipative_solver(
        args...; kwargs..., two_dimensional_features=b.two_dimensional_features
    )
end

"""
$TYPEDSIGNATURES

Generate baseline policies for the dynamic vehicle scheduling benchmark.
Returns a tuple containing:
- `lazy`: A policy that dispatches vehicles only when they are ready
- `greedy`: A policy that dispatches vehicles to the nearest customer
"""
function Utils.generate_policies(b::DynamicVehicleSchedulingBenchmark)
    lazy = Policy(
        "Lazy",
        "Lazy policy that dispatches vehicles only when they are ready.",
        lazy_policy,
    )
    greedy = Policy(
        "Greedy",
        "Greedy policy that dispatches vehicles to the nearest customer.",
        greedy_policy,
    )
    return (lazy, greedy)
end

"""
$TYPEDSIGNATURES

Generate a statistical model for the dynamic vehicle scheduling benchmark.
The model is a simple linear chain with a single dense layer that maps features to a scalar output.
The input dimension depends on whether two-dimensional features are used (2 features) or not (27 features).
"""
function Utils.generate_statistical_model(
    b::DynamicVehicleSchedulingBenchmark; seed=nothing
)
    Random.seed!(seed)
    return Chain(Dense((b.two_dimensional_features ? 2 : 27) => 1), vec)
end

export DynamicVehicleSchedulingBenchmark

end
