module Utils

using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Flux: softplus
using HiGHS: HiGHS
using JuMP: Model
using LinearAlgebra: dot
using Random: Random, MersenneTwister, AbstractRNG
using SCIP: SCIP
using SimpleWeightedGraphs: SimpleWeightedDiGraph
using StatsBase: StatsBase
using Statistics: mean

include("data_sample.jl")
include("maximizers.jl")
include("environment.jl")
include("policy.jl")
include("interface/abstract_benchmark.jl")
include("interface/static_benchmark.jl")
include("interface/stochastic_benchmark.jl")
include("interface/dynamic_benchmark.jl")
include("grid_graph.jl")
include("misc.jl")
include("model_builders.jl")

export DataSample, Policy
export evaluate_policy!
export TopKMaximizer, one_hot_argmax

export AbstractEnvironment, get_seed, is_terminated, observe, reset!, step!

export AbstractBenchmark,
    AbstractStaticBenchmark, AbstractStochasticBenchmark, AbstractDynamicBenchmark
export ExogenousStochasticBenchmark,
    EndogenousStochasticBenchmark, ExogenousDynamicBenchmark, EndogenousDynamicBenchmark
export generate_instance, generate_sample, generate_dataset
export generate_statistical_model, generate_maximizer
export generate_scenario, generate_context
export generate_environment, generate_environments
export SampleAverageApproximation
export generate_baseline_policies
export generate_anticipative_solver, generate_parametric_anticipative_solver
export is_minimization_problem

export has_visualization, plot_instance, plot_solution, plot_trajectory, animate_trajectory
export compute_gap
export grid_graph, get_path, path_to_matrix
export neg_tensor, squeeze_last_dims, average_tensor
export scip_model, highs_model
export objective_value
export is_exogenous, is_endogenous

end
