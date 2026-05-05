@testset "Subset selection" begin
    using DecisionFocusedLearningBenchmarks

    n = 25
    k = 5

    b_identity = SubsetSelectionBenchmark(; n=n, k=k)
    b = SubsetSelectionBenchmark(; n=n, k=k, identity_mapping=false)

    io = IOBuffer()
    show(io, b)
    @test String(take!(io)) == "SubsetSelectionBenchmark(n=25, k=5)"

    dataset = generate_dataset(b_identity, 50)
    dataset2 = generate_dataset(b, 50)
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)

    for (i, sample) in enumerate(dataset)
        x = sample.x
        θ_true = sample.θ
        y_true = sample.y
        @test size(x) == (n,)
        @test length(θ_true) == n
        @test length(y_true) == n
        @test isempty(sample.context)
        @test all(y_true .== maximizer(θ_true))

        # Features and true weights should be equal
        @test all(θ_true .== x)

        θ = model(x)
        @test length(θ) == n

        y = maximizer(θ)
        @test length(y) == n
        @test sum(y) == k
    end

    gap = compute_gap(b, dataset[1:5], model, maximizer)
    @test isfinite(gap)
    @test gap >= 0

    @testset "Plots" begin
        using Plots
        @test has_visualization(b_identity)
        fig1 = plot_context(b_identity, dataset[1])
        @test fig1 isa Plots.Plot
        fig2 = plot_sample(b_identity, dataset[1])
        @test fig2 isa Plots.Plot
        fig3 = plot_sample(b_identity, dataset[1], dataset[2].y)
        @test fig3 isa Plots.Plot
    end
end
