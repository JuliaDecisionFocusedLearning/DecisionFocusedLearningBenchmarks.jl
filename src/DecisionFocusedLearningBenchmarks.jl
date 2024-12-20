module DecisionFocusedLearningBenchmarks

using DataDeps
using Requires: @require

function __init__()
    # Register the Warcraft dataset
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    register(
        DataDep(
            "warcraft",
            "This is the warcraft dataset",
            "http://cermics.enpc.fr/~bouvierl/warcraft_TP/data.zip";
            post_fetch_method=unpack,
        ),
    )

    # Gurobi setup
    @info "If you have Gurobi installed and want to use it, make sure to `using Gurobi` in order to enable it."
    @require Gurobi = "2e9cd046-0924-5485-92f1-d5272153d98b" include("gurobi_setup.jl")
    return nothing
end

include("Utils/Utils.jl")

include("Argmax/Argmax.jl")
include("Ranking/Ranking.jl")
include("SubsetSelection/SubsetSelection.jl")
include("Warcraft/Warcraft.jl")
include("FixedSizeShortestPath/FixedSizeShortestPath.jl")
include("PortfolioOptimization/PortfolioOptimization.jl")

include("StochasticVehicleScheduling/StochasticVehicleScheduling.jl")

using .Utils
using .Argmax
using .Ranking
using .SubsetSelection
using .Warcraft
using .FixedSizeShortestPath
using .PortfolioOptimization
using .StochasticVehicleScheduling

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
export StochasticVehicleSchedulingBenchmark

end # module DecisionFocusedLearningBenchmarks
