module Utils

using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Flux: softplus
using LinearAlgebra: dot
using SimpleWeightedGraphs: SimpleWeightedDiGraph

include("data_sample.jl")
include("interface.jl")
include("grid_graph.jl")
include("misc.jl")

export DataSample

export AbstractBenchmark
export generate_dataset,
    generate_statistical_model, generate_maximizer, plot_data, compute_gap
export maximizer_kwargs
export grid_graph, get_path, path_to_matrix
export neg_tensor, squeeze_last_dims, average_tensor

end
