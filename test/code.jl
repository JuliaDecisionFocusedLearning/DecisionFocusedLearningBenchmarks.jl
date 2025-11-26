@testset "Aqua" begin
    using Aqua
    Aqua.test_all(
        DecisionFocusedLearningBenchmarks;
        ambiguities=false,
        deps_compat=(check_extras = false),
    )
end

@testset "JET" begin
    using JET
    JET.test_package(
        DecisionFocusedLearningBenchmarks;
        target_modules=[DecisionFocusedLearningBenchmarks],
    )
end

@testset "JuliaFormatter" begin
    using JuliaFormatter
    @test JuliaFormatter.format(
        DecisionFocusedLearningBenchmarks; verbose=false, overwrite=false
    )
end

@testset "Documenter" begin
    using Documenter
    Documenter.doctest(DecisionFocusedLearningBenchmarks)
end
