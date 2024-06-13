module InferOptBenchmarks

using HiGHS
using InferOpt

include("Utils/Utils.jl")

include("Warcraft/Warcraft.jl")
include("FixedSizeShortestPath/FixedSizeShortestPath.jl")
include("PortfolioOptimization/PortfolioOptimization.jl")
include("SubsetSelection/SubsetSelection.jl")

using .Utils

export AbstractBenchmark, generate_dataset, generate_statistical_model, generate_maximizer

end # module InferOptBenchmarks
