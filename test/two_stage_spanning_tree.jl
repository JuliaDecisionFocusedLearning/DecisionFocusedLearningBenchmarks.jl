@testset "Two-Stage Spanning Tree" begin
    using Graphs: ne
    b = TwoStageSpanningTreeBenchmark(; n=3, m=3, nb_scenarios=5)

    nb_features = 2 + 7 * length(0.0:0.1:1.0)  # 79

    # Unlabeled dataset: 2 instances × 3 evaluation scenarios each
    dataset = generate_dataset(b, 2; nb_scenarios=3, seed=1)
    @test length(dataset) == 6
    @test all(s -> s.y === nothing, dataset)
    @test all(s -> hasproperty(s.extra, :scenario), dataset)
    @test all(s -> size(s.x) == (nb_features, ne(s.instance.graph)), dataset)

    # Instance has nb_scenarios feature scenarios embedded
    @test size(dataset[1].instance.second_stage_costs, 2) == 5

    # Labeled dataset with anticipative solver
    anticipative = generate_anticipative_solver(b)
    policy =
        (sample, scenarios) -> [
            DataSample(;
                sample.context...,
                x=sample.x,
                y=anticipative(ξ; sample.context...),
                extra=(; scenario=ξ),
            ) for ξ in scenarios
        ]
    labeled = generate_dataset(b, 2; nb_scenarios=3, target_policy=policy, seed=1)
    @test length(labeled) == 6
    @test all(s -> s.y isa BitVector, labeled)

    # Maximizer
    model = generate_statistical_model(b; seed=1)
    maximizer = generate_maximizer(b)
    for sample in labeled[1:3]
        θ = model(sample.x)
        y = maximizer(θ; sample.context...)
        @test y isa BitVector
        @test length(y) == ne(sample.instance.graph)
    end

    # Objective value (should be non-negative for valid solutions)
    for sample in labeled[1:3]
        obj = objective_value(b, sample)
        @test isfinite(obj)
        @test obj >= 0
    end

    # Gap (should be finite)
    gap = compute_gap(b, labeled, model, maximizer)
    @test isfinite(gap)
end
