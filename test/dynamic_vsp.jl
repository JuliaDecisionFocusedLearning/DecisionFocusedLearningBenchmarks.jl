@testset "DVSP" begin
    using DecisionFocusedLearningBenchmarks.DynamicVehicleScheduling
    using Statistics: mean

    b = DynamicVehicleSchedulingBenchmark(; two_dimensional_features=true)
    b2 = DynamicVehicleSchedulingBenchmark(; two_dimensional_features=false)

    @test is_exogenous(b)
    @test !is_endogenous(b)

    environments = generate_environments(b, 10; seed=0)

    env = environments[1]
    get_seed(env)

    policies = generate_baseline_policies(b)
    lazy = policies[1]
    greedy = policies[2]

    d = evaluate_policy!(lazy, env, 1; seed=0)[2]

    r_lazy, d = evaluate_policy!(lazy, environments, 10)
    r_greedy, d = evaluate_policy!(greedy, environments, 10)

    @test mean(r_lazy) <= mean(r_greedy)

    env = environments[1]
    scenario = env.scenario
    v, y = generate_anticipative_solution(b, env, scenario; nb_epochs=2, reset_env=true)

    maximizer = generate_maximizer(b)

    x, instance = observe(env)
    model = generate_statistical_model(b)
    θ = model(x)
    y = maximizer(θ; instance)

    environments2 = generate_environments(b2, 10; seed=0)
    env2 = environments2[1]
    x2, instance2 = observe(env2)
    model2 = generate_statistical_model(b2)
    θ2 = model2(x2)
    y2 = maximizer(θ2; instance=instance2)
    @test size(x, 1) == 2
    @test size(x2, 1) == 27

    anticipative_value, solution = generate_anticipative_solution(b, env; reset_env=true)
    reset!(env; reset_rng=true)
    cost = sum(step!(env, sample.y) for sample in solution)
    cost2 = sum(sample.reward for sample in solution)
    @test isapprox(cost, anticipative_value; atol=1e-5)
    @test isapprox(cost, cost2; atol=1e-5)
end

@testset "DVSP - generate_dataset with environments" begin
    using DecisionFocusedLearningBenchmarks.DynamicVehicleScheduling

    b = DynamicVehicleSchedulingBenchmark(; two_dimensional_features=true)
    envs = generate_environments(b, 5; seed=0)
    policies = generate_baseline_policies(b)
    lazy = policies[1]

    # target_policy takes env -> Vector{DataSample} (full trajectory)
    target_policy = env -> evaluate_policy!(lazy, env)[2]

    # vector-of-environments overload
    dataset = generate_dataset(b, envs; target_policy=target_policy)
    @test dataset isa Vector{DataSample}
    @test !isempty(dataset)
    @test all(!isnothing(s.x) for s in dataset)
    @test all(!isnothing(s.y) for s in dataset)

    # count-based wrapper
    dataset2 = generate_dataset(b, 3; seed=1, target_policy=target_policy)
    @test dataset2 isa Vector{DataSample}
    @test !isempty(dataset2)

    # seed keyword is forwarded: same seed → same dataset
    dataset3a = generate_dataset(b, 3; seed=42, target_policy=target_policy)
    dataset3b = generate_dataset(b, 3; seed=42, target_policy=target_policy)
    @test length(dataset3a) == length(dataset3b)
end
