@testitem "DVSP" begin
    using DecisionFocusedLearningBenchmarks.DynamicVehicleScheduling
    using Statistics: mean

    b = DynamicVehicleSchedulingBenchmark(; two_dimensional_features=true)

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
end
