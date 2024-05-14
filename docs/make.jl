using Documenter
using InferOptBenchmarks

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

makedocs(;
    modules=[InferOptBenchmarks, InferOptBenchmarks.Warcraft],
    authors="Members of JuliaDecisionFocusedLearning",
    sitename="InferOptBenchmarks.jl",
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",  #
        "API reference" => [
            "interface.md",
            "warcraft.md",  #
        ],
    ],
)

deploydocs(;
    repo="github.com/JuliaDecisionFocusedLearning/InferOptBenchmarks.jl", devbranch="main"
)
