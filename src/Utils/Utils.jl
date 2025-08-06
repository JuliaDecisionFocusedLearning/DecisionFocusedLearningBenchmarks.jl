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
include("interface.jl")
include("grid_graph.jl")
include("misc.jl")
include("model_builders.jl")
include("maximizers.jl")

export DataSample

export AbstractBenchmark, AbstractStochasticBenchmark, AbstractDynamicBenchmark
export generate_dataset,
    generate_statistical_model,
    generate_maximizer,
    generate_sample,
    generate_scenario,
    generate_scenario_generator,
    generate_anticipative_solver,
    generate_environment,
    generate_environments
export generate_anticipative_solution
export plot_data, compute_gap
export maximizer_kwargs
export grid_graph, get_path, path_to_matrix
export neg_tensor, squeeze_last_dims, average_tensor
export scip_model, highs_model
export objective_value
export is_exogenous, is_endogenous

export TopKMaximizer

end
