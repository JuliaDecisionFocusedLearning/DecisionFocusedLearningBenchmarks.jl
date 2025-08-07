module Utils

using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Flux: softplus
using HiGHS: HiGHS
using JuMP: Model
using LinearAlgebra: dot
using Random: Random, MersenneTwister
using SCIP: SCIP
using SimpleWeightedGraphs: SimpleWeightedDiGraph
using StatsBase: StatsBase
using Statistics: mean

include("data_sample.jl")
include("maximizers.jl")
include("environment.jl")
include("policy.jl")
include("interface.jl")
include("grid_graph.jl")
include("misc.jl")
include("model_builders.jl")

export DataSample, Policy
export run_policy!
export TopKMaximizer

export AbstractEnv, get_seed, is_terminated, observe, reset!, step!

export AbstractBenchmark, AbstractStochasticBenchmark, AbstractDynamicBenchmark
export generate_sample, generate_dataset
export generate_statistical_model, generate_maximizer
export generate_scenario
export generate_environment, generate_environments
export generate_policies
export generate_anticipative_solution

export plot_data, compute_gap
export maximizer_kwargs
export grid_graph, get_path, path_to_matrix
export neg_tensor, squeeze_last_dims, average_tensor
export scip_model, highs_model
export objective_value
export is_exogenous, is_endogenous

end
