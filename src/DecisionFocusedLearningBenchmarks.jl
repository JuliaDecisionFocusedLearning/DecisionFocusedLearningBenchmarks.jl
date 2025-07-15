module DecisionFocusedLearningBenchmarks

using DataDeps
using Requires: @require

function _euro_neurips_unpack(local_filepath)
    directory = dirname(local_filepath)
    unpack(local_filepath)
    # Move instances and delete the rest
    for filepath in readdir(
        joinpath(directory, "euro-neurips-vrp-2022-quickstart-main", "instances"); join=true
    )
        if endswith(filepath, ".txt")
            mv(filepath, joinpath(directory, basename(filepath)))
        end
    end
    rm(joinpath(directory, "euro-neurips-vrp-2022-quickstart-main"); recursive=true)
    return nothing
end

function __init__()
    # Register the Warcraft dataset
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    register(
        DataDep(
            "warcraft",
            "Warcraft shortest path dataset",
            "http://cermics.enpc.fr/~bouvierl/warcraft_TP/data.zip";
            post_fetch_method=unpack,
        ),
    )

    register(
        DataDep(
            "dvrptw",
            "EURO-NeurIPS challenge 2022 dataset for the dynamic vehicle routing problem with time windows",
            "https://github.com/ortec/euro-neurips-vrp-2022-quickstart/archive/refs/heads/main.zip";
            post_fetch_method=_euro_neurips_unpack,
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
include("DynamicVehicleScheduling/DynamicVehicleScheduling.jl")
include("DynamicAssortment/DynamicAssortment.jl")

using .Utils
using .Argmax
using .Ranking
using .SubsetSelection
using .Warcraft
using .FixedSizeShortestPath
using .PortfolioOptimization
using .StochasticVehicleScheduling
using .DynamicVehicleScheduling
using .DynamicAssortment

# Interface
export AbstractBenchmark, AbstractStochasticBenchmark, AbstractDynamicBenchmark, DataSample

export generate_sample, generate_dataset, generate_environments
export generate_scenario_generator, generate_anticipative_solver
export generate_statistical_model
export generate_maximizer, maximizer_kwargs

export objective_value
export plot_data, plot_instance, plot_solution
export compute_gap

# Export all benchmarks
export ArgmaxBenchmark
export RankingBenchmark
export SubsetSelectionBenchmark
export WarcraftBenchmark
export FixedSizeShortestPathBenchmark
export PortfolioOptimizationBenchmark
export StochasticVehicleSchedulingBenchmark
export DVSPBenchmark
export DynamicAssortmentBenchmark

end # module DecisionFocusedLearningBenchmarks
