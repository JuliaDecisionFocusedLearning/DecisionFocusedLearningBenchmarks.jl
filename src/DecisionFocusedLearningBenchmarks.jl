module DecisionFocusedLearningBenchmarks

using DataDeps
using Requires: @require

function __init__()
    function _euro_neurips_unpack(local_filepath)
        directory = dirname(local_filepath)
        unpack(local_filepath)
        # Move instances and delete the rest
        for filepath in readdir(
            joinpath(directory, "euro-neurips-vrp-2022-quickstart-main", "instances");
            join=true,
        )
            if endswith(filepath, ".txt")
                mv(filepath, joinpath(directory, basename(filepath)))
            end
        end
        rm(joinpath(directory, "euro-neurips-vrp-2022-quickstart-main"); recursive=true)
        return nothing
    end
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
include("Argmax2D/Argmax2D.jl")
include("Ranking/Ranking.jl")
include("SubsetSelection/SubsetSelection.jl")
include("Warcraft/Warcraft.jl")
include("FixedSizeShortestPath/FixedSizeShortestPath.jl")
include("PortfolioOptimization/PortfolioOptimization.jl")
include("StochasticVehicleScheduling/StochasticVehicleScheduling.jl")
include("DynamicVehicleScheduling/DynamicVehicleScheduling.jl")
include("DynamicAssortment/DynamicAssortment.jl")

using .Utils

# Interface
export AbstractBenchmark, AbstractStochasticBenchmark, AbstractDynamicBenchmark, DataSample
export AbstractEnvironment, get_seed, is_terminated, observe, reset!, step!

export Policy, evaluate_policy!

export generate_sample, generate_dataset, generate_environments, generate_environment
export generate_scenario
export generate_policies
export generate_statistical_model
export generate_maximizer
export generate_anticipative_solution
export is_exogenous, is_endogenous

export objective_value
export plot_data, plot_instance, plot_solution
export compute_gap

# Export all benchmarks
using .Argmax
using .Argmax2D
using .Ranking
using .SubsetSelection
using .Warcraft
using .FixedSizeShortestPath
using .PortfolioOptimization
using .StochasticVehicleScheduling
using .DynamicVehicleScheduling
using .DynamicAssortment

export Argmax2DBenchmark
export ArgmaxBenchmark
export DynamicAssortmentBenchmark
export DynamicVehicleSchedulingBenchmark
export FixedSizeShortestPathBenchmark
export PortfolioOptimizationBenchmark
export RankingBenchmark
export StochasticVehicleSchedulingBenchmark
export SubsetSelectionBenchmark
export WarcraftBenchmark

end # module DecisionFocusedLearningBenchmarks
