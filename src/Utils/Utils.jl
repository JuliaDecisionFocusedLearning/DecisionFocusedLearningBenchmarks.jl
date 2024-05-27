module Utils

using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES

include("dataset.jl")
include("interface.jl")

export InferOptDataset

export AbstractBenchmark
export generate_dataset, generate_statistical_model, generate_maximizer

end
