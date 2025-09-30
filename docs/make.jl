using Documenter
using DecisionFocusedLearningBenchmarks
using Literate

md_dir = joinpath(@__DIR__, "src")
tutorial_dir = joinpath(@__DIR__, "src", "tutorials")
benchmarks_dir = joinpath(@__DIR__, "src", "benchmarks")
api_dir = joinpath(@__DIR__, "src", "api")

tutorial_files = readdir(tutorial_dir)
md_tutorial_files = [split(file, ".")[1] * ".md" for file in tutorial_files]
benchmark_files = [joinpath("benchmarks", e) for e in readdir(benchmarks_dir)]

include_tutorial = true

if include_tutorial
    for file in tutorial_files
        filepath = joinpath(tutorial_dir, file)
        Literate.markdown(filepath, md_dir; documenter=true, execute=false)
    end
end

makedocs(;
    modules=[DecisionFocusedLearningBenchmarks],
    authors="Members of JuliaDecisionFocusedLearning",
    sitename="DecisionFocusedLearningBenchmarks.jl",
    format=Documenter.HTML(; size_threshold=typemax(Int)),
    pages=[
        "Home" => [
            "Getting started" => "index.md",
            "Understanding Benchmark Interfaces" => "benchmark_interfaces.md",
        ],
        "Tutorials" => include_tutorial ? md_tutorial_files : [],
        "Benchmark problems list" => benchmark_files,
        "API reference" => "api.md",
    ],
)

if include_tutorial
    for file in md_tutorial_files
        filepath = joinpath(md_dir, file)
        rm(filepath)
    end
end

deploydocs(;
    repo="github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningBenchmarks.jl",
    devbranch="main",
)
