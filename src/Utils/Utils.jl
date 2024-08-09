module Utils

using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Flux: softplus
using SimpleWeightedGraphs: SimpleWeightedDiGraph

include("dataset.jl")
include("interface.jl")
include("grid_graph.jl")
include("misc.jl")

export InferOptDataset

export AbstractBenchmark
export generate_dataset,
    generate_statistical_model, generate_maximizer, plot_data, compute_gap
export grid_graph, get_path, path_to_matrix
export neg_tensor, squeeze_last_dims, average_tensor

end
