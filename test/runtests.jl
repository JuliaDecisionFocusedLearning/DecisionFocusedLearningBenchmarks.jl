using Test
using DecisionFocusedLearningBenchmarks
using Random

@testset "DecisionFocusedLearningBenchmarks tests" begin
    @testset "Code quality" begin
        include("code.jl")
    end

    include("utils.jl")

    include("argmax.jl")
    include("argmax_2d.jl")
    include("ranking.jl")
    include("subset_selection.jl")
    include("fixed_size_shortest_path.jl")
    include("warcraft.jl")
    include("vsp.jl")
    include("portfolio_optimization.jl")

    @testset "Dynamic Vehicle Scheduling Problem" begin
        include("dynamic_vsp.jl")
        include("dynamic_vsp_plots.jl")
    end
    include("dynamic_assortment.jl")
end
