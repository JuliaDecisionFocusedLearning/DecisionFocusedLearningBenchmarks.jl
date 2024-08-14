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
