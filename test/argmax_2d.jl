@testset "Argmax2D" begin
    using DecisionFocusedLearningBenchmarks
    using Plots

    nb_features = 5
    b = Argmax2DBenchmark(; nb_features=nb_features)

    io = IOBuffer()
    show(io, b)
    @test String(take!(io)) == "Argmax2DBenchmark(nb_features=5)"

    dataset = generate_dataset(b, 50)
    model, ps, st = generate_statistical_model(b, StableRNG(0))
    maximizer = generate_maximizer(b)

    gap = compute_gap(b, dataset, model, ps, st, maximizer)
    @test gap >= 0

    @test has_visualization(b)
    figure = plot_sample(b, dataset[1])
    @test figure isa Plots.Plot
    figure2 = plot_context(b, dataset[1])
    @test figure2 isa Plots.Plot
    figure3 = plot_sample(b, DataSample(dataset[1]; y=dataset[2].y))
    @test figure3 isa Plots.Plot

    for (i, sample) in enumerate(dataset)
        x = sample.x
        θ_true = sample.θ
        y_true = sample.y
        instance = sample.instance
        @test length(x) == nb_features
        @test length(θ_true) == 2
        @test length(y_true) == 2
        @test !isnothing(instance)
        @test instance isa Vector{Vector{Float64}}
        @test all(length(vertex) == 2 for vertex in instance)
        @test y_true in instance
        @test y_true == maximizer(θ_true; instance=instance)

        θ, _ = model(x, ps, Lux.testmode(st))
        @test length(θ) == 2

        y = maximizer(θ; instance=instance)
        @test length(y) == 2
        @test y in instance
    end
end
