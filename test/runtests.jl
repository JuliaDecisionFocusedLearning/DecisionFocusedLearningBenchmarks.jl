using Aqua: Aqua
using Documenter: Documenter
using JET: JET
using JuliaFormatter: JuliaFormatter
using InferOptBenchmarks
using Test
using Zygote

@testset verbose = true "InferOptBenchmarks" begin
    @testset verbose = true "Formalities" begin
        @testset "Aqua" begin
            Aqua.test_all(
                InferOptBenchmarks;
                ambiguities = false,
                deps_compat = (check_extras = false),
            )
        end
        @testset "JET" begin
            JET.test_package(InferOptBenchmarks; target_defined_modules = true)
        end
        @testset "JuliaFormatter" begin
            @test JuliaFormatter.format(
                InferOptBenchmarks;
                verbose = false,
                overwrite = false,
            )
        end
        @testset "Documenter" begin
            Documenter.doctest(InferOptBenchmarks)
        end
    end

    @testset "Warcraft" begin
        include("WarcraftTest/WarcraftTest.jl")
    end
end
