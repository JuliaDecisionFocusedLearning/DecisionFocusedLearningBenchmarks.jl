@testitem "Argmax" begin
    using DecisionFocusedLearningBenchmarks

    instance_dim = 10
    nb_features = 5

    b = ArgmaxBenchmark(; instance_dim=instance_dim, nb_features=nb_features)

    io = IOBuffer()
    show(io, b)
    @test String(take!(io)) == "ArgmaxBenchmark(instance_dim=10, nb_features=5)"

    dataset = generate_dataset(b, 50)
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)

    gap = compute_gap(b, dataset, model, maximizer)
    @test gap >= 0

    for (i, sample) in enumerate(dataset)
        (; x, θ_true, y_true) = sample
        @test size(x) == (nb_features, instance_dim)
        @test length(θ_true) == instance_dim
        @test length(y_true) == instance_dim
        @test isnothing(sample.instance)
        @test all(y_true .== maximizer(θ_true))

        θ = model(x)
        @test length(θ) == instance_dim

        y = maximizer(θ)
        @test length(y) == instance_dim
    end
end
