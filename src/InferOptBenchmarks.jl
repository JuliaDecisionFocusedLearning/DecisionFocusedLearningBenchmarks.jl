module InferOptBenchmarks

using DataDeps
using HiGHS
using InferOpt
# using ZipFile

function __init__()
    return register(
        DataDep(
            "warcraft",
            "This is the warcraft dataset",
            "http://cermics.enpc.fr/~bouvierl/warcraft_TP/data.zip";
            post_fetch_method=unpack,
        ),
    )
end

# function unzip(file, exdir="")
#     fileFullPath = isabspath(file) ? file : joinpath(pwd(), file)
#     basePath = dirname(fileFullPath)
#     outPath = (exdir == "" ? basePath : (isabspath(exdir) ? exdir : joinpath(pwd(), exdir)))
#     isdir(outPath) ? "" : mkdir(outPath)
#     zarchive = ZipFile.Reader(fileFullPath)
#     for f in zarchive.files
#         fullFilePath = joinpath(outPath, f.name)
#         if occursin("MACOS", f.name)
#             continue
#         end
#         if (endswith(f.name, "/") || endswith(f.name, "\\"))
#             isdir(fullFilePath) || mkdir(fullFilePath)
#         else
#             write(fullFilePath, read(f))
#         end
#     end
#     return close(zarchive)
# end

include("Utils/Utils.jl")

include("Warcraft/Warcraft.jl")
include("FixedSizeShortestPath/FixedSizeShortestPath.jl")
include("PortfolioOptimization/PortfolioOptimization.jl")
include("SubsetSelection/SubsetSelection.jl")

using .Utils

export AbstractBenchmark, generate_dataset, generate_statistical_model, generate_maximizer
export unzip

end # module InferOptBenchmarks
