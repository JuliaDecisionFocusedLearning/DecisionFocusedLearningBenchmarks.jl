@testitem "DVSP" begin
    using DecisionFocusedLearningBenchmarks.DynamicVehicleScheduling

    b = DynamicVehicleSchedulingBenchmark(; two_dimensional_features=true)
    dataset = generate_dataset(b, 10)
    environments = generate_environments(b, dataset)

    env = environments[1]
    get_seed(env)

    policies = generate_policies(b)
    lazy = policies[1]
    greedy = policies[2]

    d = run_policy!(lazy, env, 1; seed=0)[2]

    r, d = run_policy!(lazy, environments, 10)
    r, d = run_policy!(greedy, environments, 10)

    env = environments[1]
    instance = dataset[1].instance
    scenario = generate_scenario(b, instance)
    v, y = generate_anticipative_solution(b, env, scenario; nb_epochs=2, reset_env=true)
end
