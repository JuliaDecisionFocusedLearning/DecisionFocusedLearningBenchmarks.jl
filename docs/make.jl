using Documenter
using DecisionFocusedLearningBenchmarks
using Literate

md_dir = joinpath(@__DIR__, "src")
tutorial_dir = joinpath(@__DIR__, "src", "tutorials")
benchmarks_dir = joinpath(@__DIR__, "src", "benchmarks")

tutorial_files = readdir(tutorial_dir)
md_tutorial_files = [split(file, ".")[1] * ".md" for file in tutorial_files]

categories = [
    "Toy problems" => "toy",
    "Static problems" => "static",
    "Stochastic problems" => "stochastic",
    "Dynamic problems" => "dynamic",
]

include_tutorial = true

if include_tutorial
    for file in tutorial_files
        filepath = joinpath(tutorial_dir, file)
        Literate.markdown(filepath, md_dir; documenter=true, execute=false)
    end
end

benchmark_sections = Pair{String,Vector{String}}[]

for (label, subdir) in categories
    dir = joinpath(benchmarks_dir, subdir)
    jl_files = filter(f -> endswith(f, ".jl"), readdir(dir))
    md_names = [splitext(f)[1] * ".md" for f in jl_files]
    for file in jl_files
        Literate.markdown(joinpath(dir, file), dir; documenter=true, execute=false)
    end
    md_paths = [joinpath("benchmarks", subdir, f) for f in md_names]
    push!(benchmark_sections, label => md_paths)
end

makedocs(;
    modules=[DecisionFocusedLearningBenchmarks],
    authors="Members of JuliaDecisionFocusedLearning",
    sitename="DecisionFocusedLearningBenchmarks.jl",
    format=Documenter.HTML(; size_threshold=typemax(Int)),
    pages=[
        "Home" => "index.md",
        "Guides" => [
            "Using benchmarks" => "using_benchmarks.md",
            "Creating custom benchmarks" => "custom_benchmarks.md",
        ],
        "Tutorials" => include_tutorial ? md_tutorial_files : [],
        "Benchmarks" => benchmark_sections,
        "API reference" => "api.md",
    ],
)

if include_tutorial
    for file in md_tutorial_files
        filepath = joinpath(md_dir, file)
        rm(filepath)
    end
end

for (_, subdir) in categories
    dir = joinpath(benchmarks_dir, subdir)
    for f in filter(f -> endswith(f, ".md"), readdir(dir))
        rm(joinpath(dir, f); force=true)
    end
end

deploydocs(;
    repo="github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningBenchmarks.jl",
    devbranch="main",
)
