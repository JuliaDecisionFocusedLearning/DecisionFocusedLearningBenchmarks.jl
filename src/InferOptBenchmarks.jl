module InferOptBenchmarks

using DataDeps
using HiGHS
using InferOpt

function __init__()
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    register(
        DataDep(
            "warcraft",
            "This is the warcraft dataset",
            "http://cermics.enpc.fr/~bouvierl/warcraft_TP/data.zip";
            post_fetch_method=unpack,
        ),
    )
    return nothing
end

include("Utils/Utils.jl")

include("Warcraft/Warcraft.jl")
include("FixedSizeShortestPath/FixedSizeShortestPath.jl")
include("PortfolioOptimization/PortfolioOptimization.jl")
include("SubsetSelection/SubsetSelection.jl")

using .Utils
using .Warcraft
using .FixedSizeShortestPath
using .PortfolioOptimization
using .SubsetSelection

# Interface
export AbstractBenchmark, InferOptDataset
export generate_dataset
export generate_statistical_model
export generate_maximizer
export plot_data
export compute_gap

# Export all benchmarks
export WarcraftBenchmark
export FixedSizeShortestPathBenchmark
export PortfolioOptimizationBenchmark
export SubsetSelectionBenchmark

end # module InferOptBenchmarks
