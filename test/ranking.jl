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
        @test isempty(sample.context)
        @test all(y_true .== maximizer(θ_true))

        θ = model(x)
        @test length(θ) == instance_dim

        y = maximizer(θ)
        @test length(y) == instance_dim
    end

    gap = compute_gap(b, dataset[1:5], model, maximizer)
    @test isfinite(gap)
    @test gap >= 0

    @testset "Plots" begin
        using Plots
        @test has_visualization(b)
        fig1 = plot_context(b, dataset[1])
        @test fig1 isa Plots.Plot
        fig2 = plot_sample(b, dataset[1])
        @test fig2 isa Plots.Plot
        fig3 = plot_sample(b, dataset[1], dataset[2].y)
        @test fig3 isa Plots.Plot
    end
end
