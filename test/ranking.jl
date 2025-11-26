@testset "Ranking" begin
    using DecisionFocusedLearningBenchmarks

    instance_dim = 10
    nb_features = 5

    b = RankingBenchmark(; instance_dim=instance_dim, nb_features=nb_features)

    io = IOBuffer()
    show(io, b)
    @test String(take!(io)) == "RankingBenchmark(instance_dim=10, nb_features=5)"

    dataset = generate_dataset(b, 50)
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)

    for (i, sample) in enumerate(dataset)
        x = sample.x
        θ_true = sample.θ
        y_true = sample.y
        @test size(x) == (nb_features, instance_dim)
        @test length(θ_true) == instance_dim
        @test length(y_true) == instance_dim
        @test length(sample.info) == 0
        @test all(y_true .== maximizer(θ_true))

        θ = model(x)
        @test length(θ) == instance_dim

        y = maximizer(θ)
        @test length(y) == instance_dim
    end
end
