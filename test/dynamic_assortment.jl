@testitem "dynamic Assortment" begin
    using DecisionFocusedLearningBenchmarks
    using Statistics: mean

    b = DynamicAssortmentBenchmark()

    @test is_endogenous(b)
    @test !is_exogenous(b)

    dataset = generate_dataset(b, 10; seed=0)
    environments = generate_environments(b, dataset)

    env = environments[1]
    get_seed(env)
    env.seed

    policies = generate_policies(b)
    expert = policies[1]
    greedy = policies[2]

    r_expert, d = evaluate_policy!(expert, environments)
    r_greedy, _ = evaluate_policy!(greedy, environments)

    @test mean(r_expert) >= mean(r_greedy)

    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)
    sample = d[1]
    x = sample.x
    θ = model(x)
    y = maximizer(θ)
end
