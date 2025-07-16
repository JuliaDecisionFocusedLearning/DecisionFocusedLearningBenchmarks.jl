@testitem "Argmax2D" begin
    using DecisionFocusedLearningBenchmarks

    nb_features = 5
    b = Argmax2DBenchmark(; nb_features=nb_features)

    io = IOBuffer()
    show(io, b)
    @test String(take!(io)) == "Argmax2DBenchmark(nb_features=5)"

    dataset = generate_dataset(b, 50)
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)

    for (i, sample) in enumerate(dataset)
        (; x, θ_true, y_true, instance) = sample
        @test length(x) == nb_features
        @test length(θ_true) == 2  # 2D vectors
        @test length(y_true) == 2  # 2D point
        @test !isnothing(sample.instance)  # instance is a polytope
        @test instance isa Vector{Vector{Float64}}  # polytope is vector of 2D points
        @test all(length(vertex) == 2 for vertex in instance)  # all vertices are 2D
        @test y_true in instance  # solution should be a vertex of the polytope
        @test y_true == maximizer(θ_true; instance=instance)

        θ = model(x)
        @test length(θ) == 2  # 2D vector

        y = maximizer(θ; instance=instance)
        @test length(y) == 2  # 2D point
        @test y in instance  # solution should be a vertex of the polytope
    end
end
