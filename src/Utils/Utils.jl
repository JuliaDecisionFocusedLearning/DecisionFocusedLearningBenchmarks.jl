module Utils

using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES

include("interface.jl")

export AbstractBenchmark
export generate_dataset, generate_statistical_model, generate_maximizer

end
