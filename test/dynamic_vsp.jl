@testitem "DVSP" begin
    using DecisionFocusedLearningBenchmarks.DynamicVehicleScheduling
    using Statistics: mean

    b = DynamicVehicleSchedulingBenchmark(; two_dimensional_features=true)
    b2 = DynamicVehicleSchedulingBenchmark(; two_dimensional_features=false)

    @test is_exogenous(b)
    @test !is_endogenous(b)

    dataset = generate_dataset(b, 10)
    environments = generate_environments(b, dataset; seed=0)

    env = environments[1]
    get_seed(env)

    policies = generate_policies(b)
    lazy = policies[1]
    greedy = policies[2]

    d = evaluate_policy!(lazy, env, 1; seed=0)[2]

    r_lazy, d = evaluate_policy!(lazy, environments, 10)
    r_greedy, d = evaluate_policy!(greedy, environments, 10)

    @test mean(r_lazy) <= mean(r_greedy)

    env = environments[1]
    instance = dataset[1].instance
    scenario = generate_scenario(b, instance)
    v, y = generate_anticipative_solution(b, env, scenario; nb_epochs=2, reset_env=true)

    maximizer = generate_maximizer(b)

    x, instance = observe(env)
    model = generate_statistical_model(b)
    θ = model(x)
    y = maximizer(θ; instance)

    dataset2 = generate_dataset(b2, 10)
    environments2 = generate_environments(b2, dataset2; seed=0)
    env2 = environments2[1]
    x2, instance2 = observe(env2)
    model2 = generate_statistical_model(b2)
    θ2 = model2(x2)
    y2 = maximizer(θ2; instance=instance2)
    @test size(x, 1) == 2
    @test size(x2, 1) == 14
end
