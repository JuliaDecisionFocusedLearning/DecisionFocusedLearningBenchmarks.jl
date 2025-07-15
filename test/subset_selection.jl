@testitem "Subset selection" begin
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
        (; x, θ_true, y_true) = sample
        @test size(x) == (n,)
        @test length(θ_true) == n
        @test length(y_true) == n
        @test isnothing(sample.instance)
        @test all(y_true .== maximizer(θ_true))

        # Features and true weights should be equal
        @test all(θ_true .== x)

        θ = model(x)
        @test length(θ) == n

        y = maximizer(θ)
        @test length(y) == n
        @test sum(y) == k
    end
end
