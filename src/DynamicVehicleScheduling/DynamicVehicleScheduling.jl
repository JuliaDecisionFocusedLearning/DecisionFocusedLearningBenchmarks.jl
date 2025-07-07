module DynamicVehicleScheduling

using ..Utils

using Base: @kwdef
using CommonRLInterface: CommonRLInterface, AbstractEnv, reset!, terminated, observe, act!
using DataDeps: @datadep_str
# using ChainRulesCore
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Graphs
using HiGHS
# using InferOpt
using IterTools: partition
using JSON
using JuMP
using Plots: plot, plot!, scatter!
using Printf: @printf
using Random: Random, AbstractRNG, MersenneTwister, seed!, randperm
using Requires: @require
using Statistics: mean, quantile

include("utils.jl")

include("abstract_policy.jl")

# static vsp stuff
include("static_vsp/instance.jl")
include("static_vsp/parsing.jl")
include("static_vsp/solution.jl")
include("static_vsp/plot.jl")

# dynamic environment
include("environment/instance.jl")
include("environment/state.jl")
include("environment/scenario.jl")
include("environment/environment.jl")
include("environment/plot.jl")

include("algorithms/prize_collecting_vsp.jl")
include("algorithms/anticipative_solver.jl")

include("learning/features.jl")
include("learning/2d_features.jl")
include("learning/dataset.jl")

include("policy/abstract_vsp_policy.jl")
include("policy/greedy_policy.jl")
include("policy/lazy_policy.jl")
include("policy/anticipative_policy.jl")
include("policy/kleopatra_policy.jl")

struct DVSPBenchmark <: AbstractDynamicBenchmark end

function Utils.generate_sample(b::DVSPBenchmark, rng::AbstractRNG)
    return DataSample(;
        instance=Instance(read_vsp_instance(readdir(datadep"dvrptw"; join=true)[1]))
    )
end

function Utils.generate_scenario_generator(::DVSPBenchmark)
    return generate_scenario
end

function Utils.generate_anticipative_solver(::DVSPBenchmark; kwargs...)
    return anticipative_solver
end

function Utils.generate_environment(::DVSPBenchmark, instance::Instance; kwargs...)
    return DVSPEnv(instance; kwargs...)
end

function Utils.generate_maximizer(::DVSPBenchmark)
    return prize_collecting_vsp
end

export DVSPBenchmark #, generate_environment # , generate_sample, generate_anticipative_solver
export run_policy!,
    GreedyVSPPolicy, LazyVSPPolicy, KleopatraVSPPolicy, AnticipativeVSPPolicy

end
