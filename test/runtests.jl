using Aqua: Aqua
using Documenter: Documenter
using Flux
using InferOpt
using InferOptBenchmarks
using JET: JET
using JuliaFormatter: JuliaFormatter
using ProgressMeter
using Test
using UnicodePlots
using Zygote

@testset verbose = true "InferOptBenchmarks" begin
    @testset verbose = true "Formalities" begin
        @testset "Aqua" begin
            Aqua.test_all(
                InferOptBenchmarks; ambiguities=false, deps_compat=(check_extras = false)
            )
        end
        @testset "JET" begin
            JET.test_package(InferOptBenchmarks; target_defined_modules=true)
        end
        @testset "JuliaFormatter" begin
            @test JuliaFormatter.format(InferOptBenchmarks; verbose=false, overwrite=false)
        end
        @testset "Documenter" begin
            Documenter.doctest(InferOptBenchmarks)
        end
    end

    @testset "Warcraft" begin
        include("WarcraftTest/WarcraftTest.jl")
    end

    @testset "Shortest paths" begin
        bench = ShortestPathBenchmark(50)
        features = get_features(bench)
        solutions = get_solutions(bench)
        maximizer = get_maximizer(bench)

        model = Chain(Dense(5, 40), softplus)
        perturbed = PerturbedAdditive(maximizer; nb_samples=10, ε=0.1)
        fyl = FenchelYoungLoss(perturbed)

        opt_state = Flux.setup(Adam(), model)
        loss_history = Float64[]
        gap_history = Float64[]
        E = 500
        @showprogress for epoch in 1:E
            loss = 0.0
            for (x, y) in zip(features, solutions)
                val, grads = Flux.withgradient(model) do m
                    θ = m(x)
                    fyl(θ, y)
                end
                loss += val
                Flux.update!(opt_state, model, grads[1])
            end
            push!(loss_history, loss ./ E)
            push!(gap_history, compute_gap(bench, model) .* 100)
        end

        println(lineplot(loss_history; title="Loss"))
        println(lineplot(gap_history; title="Gap"))
    end
end
