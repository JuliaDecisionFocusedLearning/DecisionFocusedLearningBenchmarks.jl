module SubsetSelection

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Flux: Chain, Dense
using Random

include("subset_selection.jl")

export SubsetSelectionBenchmark
export generate_dataset, generate_statistical_model, generate_maximizer

end
