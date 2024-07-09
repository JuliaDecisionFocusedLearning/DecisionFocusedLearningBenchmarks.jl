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

export AbstractBenchmark, generate_dataset, generate_statistical_model, generate_maximizer

end # module InferOptBenchmarks
