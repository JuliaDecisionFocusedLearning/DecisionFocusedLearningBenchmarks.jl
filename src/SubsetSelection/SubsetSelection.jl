module SubsetSelection

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Flux: Chain, Dense
using Random

include("subset_selection.jl")

export SubsetSelectionBenchmark

end
