module Utils

using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Flux: softplus
using HiGHS: HiGHS
using JuMP: Model
using LinearAlgebra: dot
using SCIP: SCIP
using SimpleWeightedGraphs: SimpleWeightedDiGraph

include("data_sample.jl")
include("interface.jl")
include("grid_graph.jl")
include("misc.jl")
include("model_builders.jl")

export DataSample

export AbstractBenchmark
export generate_dataset,
    generate_statistical_model, generate_maximizer, plot_data, compute_gap
export maximizer_kwargs
export grid_graph, get_path, path_to_matrix
export neg_tensor, squeeze_last_dims, average_tensor
export scip_model, highs_model
export objective_value

end
