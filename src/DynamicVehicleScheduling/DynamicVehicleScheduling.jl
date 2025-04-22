module DynamicVehicleScheduling

using ..Utils

using Base: @kwdef
# using ChainRulesCore
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Graphs
using HiGHS
using InferOpt
using IterTools: partition
using JSON
using JuMP
using Plots: plot, plot!, scatter!
using Printf: @printf
using Random: AbstractRNG, MersenneTwister, seed!, randperm
using Requires: @require
using Statistics: mean, quantile

include("utils.jl")
include("dynamic_config.jl")
include("abstract_policy.jl")

# Dynamic Vehicle Scheduling
include("DynamicVSP/utils.jl")

include("DynamicVSP/vsp/instance.jl")
include("DynamicVSP/vsp/parsing.jl")
include("DynamicVSP/vsp/solution.jl")
include("DynamicVSP/vsp/plot.jl")

include("DynamicVSP/environment/state.jl")
include("DynamicVSP/environment/environment.jl")
include("DynamicVSP/environment/plot.jl")

include("DynamicVSP/algorithms/prize_collecting_vsp.jl")
include("DynamicVSP/algorithms/anticipative_solver.jl")

include("DynamicVSP/learning/features.jl")
include("DynamicVSP/learning/2d_features.jl")
include("DynamicVSP/learning/dataset.jl")

include("DynamicVSP/policy/abstract_vsp_policy.jl")
include("DynamicVSP/policy/greedy_policy.jl")
include("DynamicVSP/policy/lazy_policy.jl")
include("DynamicVSP/policy/anticipative_policy.jl")
include("DynamicVSP/policy/kleopatra_policy.jl")

export highs_model, filtered_readdir

export solve_hindsight_problem

export AbstractDynamicPolicy, BasicDynamicPolicy

export GreedyPolicy, LazyPolicy, RandomPolicy, Kleopatra

export run_policy

export compute_features,
    compute_2D_features, compute_critic_features, compute_critic_2D_features, load_dataset

export VSPInstance,
    read_vsp_instance, start_time, env_routes_from_state_routes, state_route_from_env_routes
export DVSPEnv, prize_collecting_vsp
export anticipative_solver
export VSPSolution
export load_VSP_dataset
export GreedyVSPPolicy,
    LazyVSPPolicy, AnticipativeVSPPolicy, run_policy!, KleopatraVSPPolicy
export plot_routes, plot_instance, plot_environment, plot_epoch
export get_state
export nb_epochs, get_epoch_indices
end
