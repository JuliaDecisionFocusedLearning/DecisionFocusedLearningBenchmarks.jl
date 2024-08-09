@testitem "Aqua" begin
    using Aqua
    Aqua.test_all(InferOptBenchmarks; ambiguities=false, deps_compat=(check_extras = false))
end

@testitem "JET" begin
    using JET
    JET.test_package(InferOptBenchmarks; target_defined_modules=true)
end

@testitem "JuliaFormatter" begin
    using JuliaFormatter
    JuliaFormatter.format(InferOptBenchmarks; verbose=false, overwrite=false)
end

@testitem "Documenter" begin
    using Documenter
    Documenter.doctest(InferOptBenchmarks)
end

# @testset verbose = true "InferOptBenchmarks" begin
#     @testset verbose = true "Formalities" begin
#         @testset "Aqua" begin
#             Aqua.test_all(
#                 InferOptBenchmarks; ambiguities=false, deps_compat=(check_extras = false)
#             )
#         end
#         @testset "JET" begin
#             JET.test_package(InferOptBenchmarks; target_defined_modules=true)
#         end
#         @testset "JuliaFormatter" begin
#             @test JuliaFormatter.format(InferOptBenchmarks; verbose=false, overwrite=false)
#         end
#         @testset "Documenter" begin
#             Documenter.doctest(InferOptBenchmarks)
#         end
#     end

#     @testset "Warcraft" begin
#         include("WarcraftTest/WarcraftTest.jl")
#     end

#     @testset "Shortest paths" begin
#         include("FixedSizeShortestPathTest/FixedSizeShortestPathTest.jl")
#     end

#     @testset "Portfolio Optimization" begin
#         include("PortfolioOptimizationTest/PortfolioOptimizationTest.jl")
#     end

#     @testset "Subset Selection" begin
#         include("SubsetSelectionTest/SubsetSelectionTest.jl")
#     end
# end
