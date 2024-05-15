module InferOptBenchmarks

using InferOpt

include("Utils/Utils.jl")
include("Warcraft/Warcraft.jl")
include("ShortestPath/ShortestPath.jl")

using .Utils

export AbstractBenchmark, generate_dataset, generate_statistical_model, generate_maximizer

end # module InferOptBenchmarks
