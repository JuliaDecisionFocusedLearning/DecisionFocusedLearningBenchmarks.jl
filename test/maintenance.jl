const maintenance = DecisionFocusedLearningBenchmarks.Maintenance

@testset "Maintenance - Benchmark Construction" begin
    # Test default constructor
    b = MaintenanceBenchmark()
    @test b.N == 2
    @test b.K == 1
    @test b.n == 3
    @test b.p == 0.2
    @test b.c_f == 10.0
    @test b.c_m == 3.0
    @test b.max_steps == 80
    @test is_exogenous(b)
    @test !is_endogenous(b)

    # Test custom constructor
    b_custom = MaintenanceBenchmark(; N=10, K=3, n=5, p=0.3, c_f=5.0, c_m=3.0, max_steps=50)
    @test b_custom.N == 10
    @test b_custom.K == 3
    @test b_custom.n == 5
    @test b_custom.p == 0.3
    @test b_custom.c_f == 5.0
    @test b_custom.c_m == 3.0
    @test b_custom.max_steps == 50

    # Test accessor functions
    @test maintenance.component_count(b) == 2
    @test maintenance.maintenance_capacity(b) == 1
    @test maintenance.degradation_levels(b) == 3
    @test maintenance.degradation_probability(b) == 0.2
    @test maintenance.failure_cost(b) == 10.0
    @test maintenance.maintenance_cost(b) == 3.0
    @test maintenance.max_steps(b) == 80
end

@testset "Maintenance - Instance Generation" begin
    b = MaintenanceBenchmark(; N=10, K=3, n=5, p=0.3, c_f=5.0, c_m=3.0, max_steps=50)
    rng = MersenneTwister(42)

    instance = maintenance.Instance(b, rng)

    # test state is randomly initialized
    state1 = maintenance.starting_state(instance)
    rng2 = MersenneTwister(43)
    instance2 = maintenance.Instance(b, rng2)
    state2 = maintenance.starting_state(instance2)
    @test state1 != state2

    # Test instance structure
    @test length(instance.starting_state) == 10 
    @test all(1.0 ≤ s ≤ 5 for s in instance.starting_state)  

    # Test accessor functions
    @test maintenance.component_count(instance) == 10
    @test maintenance.maintenance_capacity(instance) == 3
    @test maintenance.degradation_levels(instance) == 5
    @test maintenance.degradation_probability(instance) == 0.3
    @test maintenance.failure_cost(instance) == 5.0
    @test maintenance.maintenance_cost(instance) == 3.0
    @test maintenance.max_steps(instance) == 50
end

@testset "Maintenance - Environment Initialization" begin
    b = MaintenanceBenchmark()
    instance = maintenance.Instance(b, MersenneTwister(42))

    env = maintenance.Environment(instance; seed=123)

    # Test initial state
    @test env.step == 1
    @test env.seed == 123
    @test !is_terminated(env)

    # Test accessor functions
    @test maintenance.component_count(env) == 2
    @test maintenance.maintenance_capacity(env) == 1
    @test maintenance.degradation_levels(env) == 3
    @test maintenance.degradation_probability(env) == 0.2
    @test maintenance.failure_cost(env) == 10.0
    @test maintenance.maintenance_cost(env) == 3.0
    @test maintenance.max_steps(env) == 80
end

@testset "Maintenance - Environment Reset" begin
    b = MaintenanceBenchmark()
    instance = maintenance.Instance(b, MersenneTwister(42))
    env = maintenance.Environment(instance; seed=123)

    # Modify environment state
    env.step = 3

    # Reset environment
    reset!(env)

    # Check reset state
    @test env.step == 1
end

@testset "Maintenance - Cost" begin
    b = MaintenanceBenchmark()
    instance = maintenance.Instance(b, MersenneTwister(42))
    env = maintenance.Environment(instance; seed=123)

    env.degradation_state = [1,1]
    @test maintenance.maintenance_cost(env, BitVector([false, false])) == 0.0 
    @test maintenance.maintenance_cost(env, BitVector([false, true])) == 3.0 
    @test maintenance.maintenance_cost(env, BitVector([true, true])) == 6.0 

    @test maintenance.degradation_cost(env) == 0.0 
    env.degradation_state = [2,2]
    @test maintenance.degradation_cost(env) == 0.0 
    env.degradation_state = [3,2]
    @test maintenance.degradation_cost(env) == 10.0 
    env.degradation_state = [3,3]
    @test maintenance.degradation_cost(env) == 20.0 
end

@testset "Maintenance - Environment Step" begin
    b = MaintenanceBenchmark()
    instance = maintenance.Instance(b, MersenneTwister(42))
    env = maintenance.Environment(instance; seed=123)

    maintenance_vect = BitVector([false, false])

    initial_step = env.step
    # Take a step
    reward = step!(env, maintenance_vect)

    # Check step progression
    @test env.step == initial_step + 1
    @test reward ≥ 0.0  # Reward should be non-negative 

    # Test termination
    for _ in 1:(maintenance.max_steps(env) - 1)
        if !is_terminated(env)
            step!(env, maintenance_vect)
        end
    end
    @test is_terminated(env)

    # Test error on terminated environment
    @test_throws AssertionError step!(env, maintenance_vect)
end

@testset "Maintenance - Observation" begin
    b = MaintenanceBenchmark()
    instance = maintenance.Instance(b, MersenneTwister(42))
    env = maintenance.Environment(instance; seed=123)
    env.degradation_state = [1,1]

    state, features = observe(env)

    @test state == [1,1]
    @test features === state

    env.degradation_state = [2,3]
    state2, _ = observe(env)

    @test state != state2  # Observations should differ after purchase
end


@testset "Maintenance - Policies" begin
    using Statistics: mean

    b = MaintenanceBenchmark()

    # Generate test data
    dataset = generate_dataset(b, 10; seed=0)
    environments = generate_environments(b, dataset)

    # Get policies
    policies = generate_policies(b)
    greedy = policies[1]

    @test greedy.name == "Greedy"

    # Test policy evaluation
    r_greedy, _ = evaluate_policy!(greedy, environments, 10)

    @test length(r_greedy) == length(environments)
    @test all(r_greedy .≥ 0.0)

    # Test policy output format
    env = environments[1]
    reset!(env)

    greedy_action = greedy(env)
    @test greedy_action isa BitVector && length(greedy_action) == 2
end


@testset "Maintenance - Model and Maximizer Integration" begin
    b = MaintenanceBenchmark()

    # Test statistical model generation
    model = generate_statistical_model(b; seed=42)
    # Test maximizer generation
    maximizer = generate_maximizer(b)

    # Test integration with sample data
    sample = generate_sample(b, MersenneTwister(42))
    @test hasfield(typeof(sample), :info)

    dataset = generate_dataset(b, 3; seed=42)
    environments = generate_environments(b, dataset)

    # Evaluate policy to get data samples
    policies = generate_policies(b)
    _, data_samples = evaluate_policy!(policies[1], environments)

    # Test model-maximizer pipeline
    sample = data_samples[1]
    x = sample.x
    θ = model(x)
    y = maximizer(θ)

    @test length(θ) == 2

    θ = [1,2]
    @test maximizer(θ) == BitVector([false, true])

    b = MaintenanceBenchmark(; N=10, K=3, n=5, p=0.3, c_f=5.0, c_m=3.0, max_steps=50)
    θ = [i for i in 1:10]
    maximizer = generate_maximizer(b)
    @test maximizer(θ) == BitVector([false, false, false, false, false, false, false, true, true, true])



    #test maximizer output
end
