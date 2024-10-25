using Documenter
using DecisionFocusedLearningBenchmarks
using Literate

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

md_dir = joinpath(@__DIR__, "src")
tutorial_dir = joinpath(@__DIR__, "src", "tutorials")
tutorial_files = readdir(tutorial_dir)
md_tutorial_files = [split(file, ".")[1] * ".md" for file in tutorial_files]

include_tutorial = false

if include_tutorial
    for file in tutorial_files
        filepath = joinpath(tutorial_dir, file)
        Literate.markdown(filepath, md_dir; documenter=true, execute=false)
    end
end

makedocs(;
    modules=[DecisionFocusedLearningBenchmarks, DecisionFocusedLearningBenchmarks.Warcraft],
    authors="Members of JuliaDecisionFocusedLearning",
    sitename="DecisionFocusedLearningBenchmarks.jl",
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Tutorials" => include_tutorial ? md_tutorial_files : [],
        "Benchmark problems list" => [
            "benchmarks/subset_selection.md",
            "benchmarks/portfolio_optimization.md",
            "benchmarks/fixed_size_shortest_path.md",
            "benchmarks/warcraft.md",
        ],
        "API reference" =>
            ["api/interface.md", "api/decision_focused.md", "api/warcraft.md"],
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
