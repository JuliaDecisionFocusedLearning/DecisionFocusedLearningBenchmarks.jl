module DynamicVehicleScheduling

using ..Utils

using Base: @kwdef
using DataDeps: @datadep_str
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Graphs
using HiGHS
using InferOpt: LinearMaximizer
using IterTools: partition
using JSON
using JuMP
using Plots: plot, plot!, scatter!
using Printf: @printf
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

function Utils.generate_environment(
    ::DynamicVehicleSchedulingBenchmark, instance::Instance, rng::AbstractRNG; kwargs...
)
    seed = rand(rng, 1:typemax(Int))
    return DVSPEnv(instance; seed)
end

function Utils.generate_maximizer(::DynamicVehicleSchedulingBenchmark)
    return LinearMaximizer(oracle; g, h)
end

function Utils.generate_scenario(b::DynamicVehicleSchedulingBenchmark, args...; kwargs...)
    return Utils.generate_scenario(args...; kwargs...)
end

function Utils.generate_anticipative_solution(
    b::DynamicVehicleSchedulingBenchmark, args...; kwargs...
)
    return anticipative_solver(
        args...; kwargs..., two_dimensional_features=b.two_dimensional_features
    )
end

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

export DynamicVehicleSchedulingBenchmark

end
