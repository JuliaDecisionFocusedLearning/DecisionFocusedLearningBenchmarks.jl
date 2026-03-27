using DecisionFocusedLearningBenchmarks
using Random
using Test

@testset "AbstractBenchmark interface" begin
    struct DummyBenchmark <: AbstractStaticBenchmark end
    b = DummyBenchmark()
    rng = MersenneTwister(1234)
    @test_throws ErrorException generate_instance(b, rng)
    @test_throws ErrorException generate_maximizer(b)
    @test_throws ErrorException generate_statistical_model(b; seed=0)
    @test !has_visualization(b)

    function DecisionFocusedLearningBenchmarks.generate_instance(
        ::DummyBenchmark, rng::AbstractRNG
    )
        return DataSample(; x=rand(rng, 5))
    end

    dataset = generate_dataset(b, 10; seed=0)
    @test length(dataset) == 10
    @test all(x -> length(x.x) == 5, dataset)

    struct DummyDynamicBenchmark <: AbstractDynamicBenchmark{true} end
    db = DummyDynamicBenchmark()
    @test_throws ErrorException compute_gap(db)
end
