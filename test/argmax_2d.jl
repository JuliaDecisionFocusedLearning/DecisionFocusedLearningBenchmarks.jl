@testitem "Argmax2D" begin
    using DecisionFocusedLearningBenchmarks
    using Plots

    nb_features = 5
    b = Argmax2DBenchmark(; nb_features=nb_features)

    io = IOBuffer()
    show(io, b)
    @test String(take!(io)) == "Argmax2DBenchmark(nb_features=5)"

    dataset = generate_dataset(b, 50)
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)

    # Test plot_data
    figure = plot_data(b, dataset[1])
    @test figure isa Plots.Plot

    for (i, sample) in enumerate(dataset)
        (; x, θ_true, y_true, instance) = sample
        @test length(x) == nb_features
        @test length(θ_true) == 2
        @test length(y_true) == 2
        @test !isnothing(sample.instance)
        @test instance isa Vector{Vector{Float64}}
        @test all(length(vertex) == 2 for vertex in instance)
        @test y_true in instance
        @test y_true == maximizer(θ_true; instance=instance)

        θ = model(x)
        @test length(θ) == 2

        y = maximizer(θ; instance=instance)
        @test length(y) == 2
        @test y in instance
    end
end
