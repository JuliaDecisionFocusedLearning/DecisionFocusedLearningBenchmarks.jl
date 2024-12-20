module DecisionFocusedLearningBenchmarks

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

include("Argmax/Argmax.jl")
include("Ranking/Ranking.jl")
include("SubsetSelection/SubsetSelection.jl")
include("Warcraft/Warcraft.jl")
include("FixedSizeShortestPath/FixedSizeShortestPath.jl")
include("PortfolioOptimization/PortfolioOptimization.jl")

using .Utils
using .Argmax
using .Ranking
using .SubsetSelection
using .Warcraft
using .FixedSizeShortestPath
using .PortfolioOptimization

# Interface
export AbstractBenchmark, DataSample
export generate_dataset
export generate_statistical_model
export generate_maximizer
export plot_data
export compute_gap

# Export all benchmarks
export ArgmaxBenchmark
export RankingBenchmark
export SubsetSelectionBenchmark
export WarcraftBenchmark
export FixedSizeShortestPathBenchmark
export PortfolioOptimizationBenchmark

end # module DecisionFocusedLearningBenchmarks
