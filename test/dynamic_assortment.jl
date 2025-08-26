@testsnippet DAPSetup begin
    const DAP = DecisionFocusedLearningBenchmarks.DynamicAssortment
end

@testitem "DynamicAssortment - Benchmark Construction" setup=[Imports, DAPSetup] begin
    # Test default constructor
    b = DynamicAssortmentBenchmark()
    @test b.N == 20
    @test b.d == 2
    @test b.K == 4
    @test b.max_steps == 80
    @test is_endogenous(b)
    @test !is_exogenous(b)

    # Test custom constructor
    b_custom = DynamicAssortmentBenchmark(N=10, d=3, K=2, max_steps=50, exogenous=true)
    @test b_custom.N == 10
    @test b_custom.d == 3
    @test b_custom.K == 2
    @test b_custom.max_steps == 50
    @test !is_endogenous(b_custom)
    @test is_exogenous(b_custom)

    # Test accessor functions
    @test DAP.item_count(b) == 20
    @test DAP.feature_count(b) == 2
    @test DAP.assortment_size(b) == 4
    @test DAP.max_steps(b) == 80
end

@testitem "DynamicAssortment - Instance Generation" setup=[Imports, DAPSetup] begin
    b = DynamicAssortmentBenchmark(N=5, d=3, K=2)
    rng = MersenneTwister(42)

    instance = DAP.Instance(b, rng)

    # Test instance structure
    @test length(instance.prices) == 6  # N items + 1 no-purchase action
    @test instance.prices[end] == 0.0  # No-purchase action has price 0
    @test all(1.0 ≤ p ≤ 10.0 for p in instance.prices[1:end-1])  # Prices in [1, 10]

    @test size(instance.features) == (3, 5)  # (d, N)
    @test all(1.0 ≤ f ≤ 10.0 for f in instance.features)  # Features in [1, 10]

    @test size(instance.starting_hype_and_saturation) == (2, 5)  # (2, N)
    @test all(1.0 ≤ f ≤ 10.0 for f in instance.starting_hype_and_saturation)

    # Test accessor functions
    @test DAP.item_count(instance) == 5
    @test DAP.feature_count(instance) == 3
    @test DAP.assortment_size(instance) == 2
    @test DAP.max_steps(instance) == 80
    @test DAP.prices(instance) == instance.prices
end

@testitem "DynamicAssortment - Environment Initialization" setup=[Imports, DAPSetup] begin
    b = DynamicAssortmentBenchmark(N=5, d=2, K=2, max_steps=10)
    instance = DAP.Instance(b, MersenneTwister(42))

    env = DAP.Environment(instance; seed=123)

    # Test initial state
    @test env.step == 1
    @test isempty(env.purchase_history)
    @test env.seed == 123
    @test !is_terminated(env)

    # Test features structure: [prices; hype_saturation; static_features]
    @test size(env.features) == (5, 5)  # (1 + 2 + d, N) = (1 + 2 + 2, 5)
    @test env.features[1, :] == instance.prices[1:end-1]  # First row is prices (excluding no-purchase)

    # Test utility computation
    @test length(env.utility) == 5  # One utility per item

    # Test accessor functions
    @test DAP.item_count(env) == 5
    @test DAP.feature_count(env) == 2
    @test DAP.assortment_size(env) == 2
    @test DAP.max_steps(env) == 10
    @test DAP.prices(env) == instance.prices
end

@testitem "DynamicAssortment - Environment Reset" setup=[Imports, DAPSetup] begin
    b = DynamicAssortmentBenchmark(N=3, d=1, K=2, max_steps=5)
    instance = DAP.Instance(b, MersenneTwister(42))
    env = DAP.Environment(instance; seed=123)

    # Modify environment state
    env.step = 3
    push!(env.purchase_history, 1, 2)
    env.features[2, 1] *= 1.5  # Modify hype

    # Reset environment
    reset!(env)

    # Check reset state
    @test env.step == 1
    @test isempty(env.purchase_history)
    @test all(env.d_features .== 0.0)

    # Features should be reset to initial values
    expected_features = vcat(
        reshape(instance.prices[1:end-1], 1, :),
        instance.starting_hype_and_saturation,
        instance.features
    )
    @test env.features ≈ expected_features
end

@testitem "DynamicAssortment - Hype Update Logic" setup=[Imports, DAPSetup] begin
    b = DynamicAssortmentBenchmark(N=5, d=1, K=2)
    instance = DAP.Instance(b, MersenneTwister(42))
    env = DAP.Environment(instance; seed=123)

    # Test hype update with no history
    hype = DAP.hype_update(env)
    @test all(hype .== 1.0)  # Should be all ones with no history

    # Test hype update with recent purchase
    push!(env.purchase_history, 2)  # Purchase item 2
    hype = DAP.hype_update(env)
    @test hype[2] ≈ 1.02  # Should increase by 0.02
    @test all(hype[i] == 1.0 for i in [1, 3, 4, 5])  # Others unchanged

    # Test hype update with older purchases
    push!(env.purchase_history, 3, 2, 1)  # More purchases
    hype = DAP.hype_update(env)
    @test hype[1] ≈ 1.02
    @test hype[2] ≈ 0.99
    @test hype[3] ≈ 0.995

    # Test with no-purchase action (item > N)
    env.purchase_history = [6]  # No-purchase action
    hype = DAP.hype_update(env)
    @test all(hype .== 1.0)  # Should not affect any item hype
end

@testitem "DynamicAssortment - Choice Probabilities" setup=[Imports, DAPSetup] begin
    b = DynamicAssortmentBenchmark(N=3, d=1, K=2)
    instance = DAP.Instance(b, MersenneTwister(42))
    env = DAP.Environment(instance; seed=123)

    # Test with full assortment
    assortment = trues(3)
    probs = DAP.choice_probabilities(env, assortment)

    @test length(probs) == 4  # 3 items + no-purchase
    @test sum(probs) ≈ 1.0  # Probabilities sum to 1
    @test all(probs .≥ 0.0)  # All probabilities non-negative

    # Test with partial assortment
    assortment = falses(3)
    assortment[1] = true
    assortment[3] = true
    probs = DAP.choice_probabilities(env, assortment)

    @test probs[2] == 0.0  # Item 2 not in assortment, so probability 0
    @test probs[1] > 0.0   # Item 1 in assortment
    @test probs[3] > 0.0   # Item 3 in assortment
    @test probs[4] > 0.0   # No-purchase always available
    @test sum(probs) ≈ 1.0

    # Test empty assortment
    assortment = falses(3)
    probs = DAP.choice_probabilities(env, assortment)
    @test all(probs[1:3] .== 0.0)  # No items available
    @test probs[4] ≈ 1.0  # Only no-purchase available
end

@testitem "DynamicAssortment - Expected Revenue" setup=[Imports, DAPSetup] begin
    b = DynamicAssortmentBenchmark(N=3, d=1, K=2)
    instance = DAP.Instance(b, MersenneTwister(42))
    env = DAP.Environment(instance; seed=123)

    # Test with full assortment
    assortment = trues(3)
    revenue = DAP.compute_expected_revenue(env, assortment)
    @test revenue ≥ 0.0

    # Test with empty assortment
    assortment = falses(3)
    revenue = DAP.compute_expected_revenue(env, assortment)
    @test revenue == 0.0  # Only no-purchase available with price 0
end

@testitem "DynamicAssortment - Environment Step" setup=[Imports, DAPSetup] begin
    b = DynamicAssortmentBenchmark(N=3, d=1, K=2, max_steps=5)
    instance = DAP.Instance(b, MersenneTwister(42))
    env = DAP.Environment(instance; seed=123)

    initial_step = env.step
    assortment = trues(3)

    # Take a step
    reward = step!(env, assortment)

    # Check step progression
    @test env.step == initial_step + 1
    @test length(env.purchase_history) == 1
    @test reward ≥ 0.0  # Reward should be non-negative (price or 0)

    # Check reward is valid price
    purchased_item = env.purchase_history[1]
    if purchased_item <= 3
        @test reward == instance.prices[purchased_item]
    else
        @test reward == 0.0  # No-purchase action
    end

    # Test termination
    for _ in 1:(DAP.max_steps(env)-1)
        if !is_terminated(env)
            step!(env, assortment)
        end
    end
    @test is_terminated(env)

    # Test error on terminated environment
    @test_throws AssertionError step!(env, assortment)
end

@testitem "DynamicAssortment - Endogenous vs Exogenous" setup=[Imports, DAPSetup] begin
    # Test endogenous environment (features change with purchases)
    b_endo = DynamicAssortmentBenchmark(N=3, d=1, K=2, exogenous=false)
    instance_endo = DAP.Instance(b_endo, MersenneTwister(42))
    env_endo = DAP.Environment(instance_endo; seed=123)

    initial_features_endo = copy(env_endo.features)
    DAP.buy_item!(env_endo, 1)

    @test env_endo.features != initial_features_endo  # Features should change
    @test any(env_endo.d_features .!= 0.0)  # Delta features should be non-zero

    # Test exogenous environment (features don't change with purchases)
    b_exo = DynamicAssortmentBenchmark(N=3, d=1, K=2, exogenous=true)
    instance_exo = DAP.Instance(b_exo, MersenneTwister(42))
    env_exo = DAP.Environment(instance_exo; seed=123)

    initial_features_exo = copy(env_exo.features)
    DAP.buy_item!(env_exo, 1)

    @test env_exo.features == initial_features_exo  # Features should not change
    @test all(env_exo.d_features .== 0.0)  # Delta features should remain zero
end

@testitem "DynamicAssortment - Observation" setup=[Imports, DAPSetup] begin
    b = DynamicAssortmentBenchmark(N=3, d=2, max_steps=10)
    instance = DAP.Instance(b, MersenneTwister(42))
    env = DAP.Environment(instance; seed=123)

    obs, info = observe(env)

    # Check observation dimensions: (d + 8, N)
    # Features: prices(1) + hype_sat(2) + static(d) + d_features(2) + delta_features(2) + step(1)
    expected_rows = 2 + 8  # d + 8 where d=2
    @test size(obs) == (expected_rows, 3)
    @test info === nothing

    @test all(-1.0 ≤ x ≤ 1.0 for x in obs)

    # Test observation changes with step
    obs1, _ = observe(env)
    DAP.buy_item!(env, 1)
    obs2, _ = observe(env)

    @test obs1 != obs2  # Observations should differ after purchase
end

@testitem "DynamicAssortment - Policies" setup=[Imports, DAPSetup] begin
    using Statistics: mean

    b = DynamicAssortmentBenchmark(N=5, d=2, K=3, max_steps=20)

    # Generate test data
    dataset = generate_dataset(b, 5; seed=0)
    environments = generate_environments(b, dataset)

    # Get policies
    policies = generate_policies(b)
    expert = policies[1]
    greedy = policies[2]

    @test expert.name == "Expert"
    @test greedy.name == "Greedy"

    # Test policy evaluation
    r_expert, d = evaluate_policy!(expert, environments)
    r_greedy, _ = evaluate_policy!(greedy, environments)

    @test length(r_expert) == length(environments)
    @test length(r_greedy) == length(environments)
    @test all(r_expert .≥ 0.0)
    @test all(r_greedy .≥ 0.0)

    # Expert should generally outperform greedy (or at least not be worse on average)
    @test mean(r_expert) ≥ mean(r_greedy)

    # Test policy output format
    env = environments[1]
    reset!(env)

    expert_action = expert(env)
    greedy_action = greedy(env)
    @test length(expert_action) == DAP.item_count(env)
    @test length(greedy_action) == DAP.item_count(env)
    @test sum(expert_action) == DAP.assortment_size(env)
    @test sum(greedy_action) == DAP.assortment_size(env)
end

@testitem "DynamicAssortment - Model and Maximizer Integration" setup=[Imports, DAPSetup] begin
    b = DynamicAssortmentBenchmark(N=4, d=3, K=2)

    # Test statistical model generation
    model = generate_statistical_model(b; seed=42)
    # Test maximizer generation
    maximizer = generate_maximizer(b)

    # Test integration with sample data
    sample = generate_sample(b, MersenneTwister(42))
    @test hasfield(typeof(sample), :instance)

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

    @test length(θ) == DAP.item_count(b)
    @test length(y) == DAP.item_count(b)
    @test sum(y) == DAP.assortment_size(b)
end
