@testitem "Aqua" begin
    using Aqua
    Aqua.test_all(
        DecisionFocusedLearningBenchmarks;
        ambiguities=false,
        deps_compat=(check_extras = false),
    )
end

@testitem "JET" begin
    using JET
    JET.test_package(DecisionFocusedLearningBenchmarks; target_defined_modules=true)
end

@testitem "JuliaFormatter" begin
    using JuliaFormatter
    @test JuliaFormatter.format(
        DecisionFocusedLearningBenchmarks; verbose=false, overwrite=false
    )
end

@testitem "Documenter" begin
    using Documenter
    Documenter.doctest(DecisionFocusedLearningBenchmarks)
end
