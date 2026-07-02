const DR = DecisionFocusedLearningBenchmarks.DynamicReplenishment

@testset "DynamicReplenishment - Benchmark Construction" begin
    b = DynamicReplenishmentBenchmark()
    @test b.N == 10
    @test b.λ == 15.0
    @test b.d == 5
    @test b.stock_inf == 0
    @test b.stock_sup == 50
    @test b.ub_same_item == 10
    @test b.delivery_delay == 3
    @test b.max_steps == 10
    # @test is_endogenous(b)
    # @test !is_exogenous(b)
    @test size(b.constraints_matrix) == (12, 10)
    @test b.quotas[1, :] == [30, 30, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    @test size(b.quotas) == (10, 12)

    b_custom = DynamicReplenishmentBenchmark(;
        N=5,
        λ=10.0,
        constraints_matrix=[1 1 1 1 1; 0 0 0 0 0; 0 0 1 1 0],
        quotas=[20, 15, 5],
        d=3,
        stock_inf=2,
        stock_sup=30,
        ub_same_item=5,
        delivery_delay=1,
        max_steps=20,
    )
    @test b_custom.N == 5
    @test b_custom.λ == 10.0
    @test b_custom.d == 3
    @test b_custom.stock_inf == 2
    @test b_custom.stock_sup == 30
    @test b_custom.ub_same_item == 5
    @test b_custom.delivery_delay == 1
    @test b_custom.max_steps == 20
    @test size(b_custom.constraints_matrix) == (8, 5)
    @test b_custom.quotas[1, :] == [20, 15, 5, 5, 5, 5, 5, 5]
    @test size(b_custom.quotas) == (20, 8)

    @test DR.item_count(b) == 10
    @test DR.feature_count(b) == 5
    @test DR.max_steps(b) == 10
    @test DR.stock_inf(b) == 0
    @test DR.stock_sup(b) == 50
    @test DR.ub_same_item(b) == 10
    @test DR.delivery_delay(b) == 3
    @test DR.poisson_arrival_rate(b) == 15.0
    @test length(DR.prices(b)) == 10
    @test all(1.0 .≤ DR.prices(b) .≤ 10.0)
    @test size(DR.features(b)) == (5, 10)
    @test length(DR.virtual_stock_cost(b)) == 10
    @test length(DR.physical_stock_cost(b)) == 10
    @test DR.nb_constraints(b) == 12
end

@testset "DynamicReplenishment - Environment Initialization" begin
    b = DynamicReplenishmentBenchmark()

    env1 = DR.Environment(b; seed=42)
    @test !is_terminated(env1)
    @test DR.item_count(env1) == 10
    @test DR.max_steps(env1) == 10
    @test length(DR.stock_ini(env1)) == 10
    @test all(0 .≤ DR.stock_ini(env1) .≤ 10)

    @test DR.current_epoch(env1) == 1
    @test env1.stock_ini == DR.stock_ini(env1)
    @test DR.stock(env1) == DR.stock_ini(env1)

    state_ini = env1.state
    @test state_ini.current_epoch == 1
    @test state_ini.stock == DR.stock_ini(env1)
    @test size(state_ini.stock_history) == (1, 10)
    @test size(state_ini.replenishment_history) == (0, 10)
    @test size(state_ini.sales_history) == (0, 10)
    @test length(state_ini.customer_history) == 0
    @test state_ini.current_cost == 0.0

    # custom environment 
    env2 = DR.Environment(b; stock_ini=fill(5, 10), seed=123)
    @test DR.stock_ini(env2) == fill(5, 10)
    @test DR.stock(env2) == fill(5, 10)
end

@testset "DynamicReplenishment - Environment Reset" begin
    b = DynamicReplenishmentBenchmark()
    env = DR.Environment(b; seed=42)

    s0 = copy(DR.stock_ini(env))
    N = DR.item_count(b)
    repl = DR.y_oracle(env, zeros(Int, N), s0)
    step!(env, repl)
    reset!(env)

    @test !is_terminated(env)
    @test DR.stock(env) == s0
    @test DR.current_epoch(env) == 1
end

@testset "DynamicReplenishment - Environment Step" begin
    b = DynamicReplenishmentBenchmark()
    env = DR.Environment(b; seed=42)
    N = DR.item_count(b)

    action = DR.y_oracle(env, zeros(Int, N), env.stock_ini)
    reward = step!(env, action)
    @test reward isa Float64
    @test DR.current_epoch(env) == 2

    # run to termination
    while !is_terminated(env)
        repl = DR.y_oracle(env, zeros(Int, N), env.state.stock)
        @test DR.is_feasible(env.state, repl)
        step!(env, repl)
    end
    @test is_terminated(env)
    @test_throws AssertionError step!(env, DR.y_oracle(env, zeros(Int, N), env.state.stock))
end

@testset "DynamicReplenishment - Feasibility" begin
    b = DynamicReplenishmentBenchmark(N=2, constraints_matrix=[1 1], quotas=[1])
    N = DR.item_count(b)
    stock_ini = [0, 0]
    env = DR.Environment(b; seed=42, stock_ini=stock_ini)
    # zero replenishment is always feasible
    @test DR.is_feasible(env.state, DR.y_oracle(env, [1, 0], env.state.stock))
    @test DR.is_feasible(env.state, DR.y_oracle(env, [0, 1], env.state.stock))
    @test DR.is_feasible(env.state, DR.y_oracle(env, [0, 0], env.state.stock))

    # replenishment exceeding quota is infeasible
    @test !DR.is_feasible(env.state, DR.y_oracle(env, [2, 0], env.state.stock))
    @test !DR.is_feasible(env.state, DR.y_oracle(env, [0, 2], env.state.stock))
    @test !DR.is_feasible(env.state, DR.y_oracle(env, [1, 1], env.state.stock))
end

@testset "DynamicReplenishment - Quota Constraints" begin
    b = DynamicReplenishmentBenchmark()
    max_quotas = DR.max_quota_per_step_per_item(b)

    @test size(max_quotas) == (DR.max_steps(b), DR.item_count(b))
    @test all(max_quotas .≥ 0)
    @test all(max_quotas .≤ DR.ub_same_item(b))
end

@testset "DynamicReplenishment - State" begin
    b = DynamicReplenishmentBenchmark()
    env = DR.Environment(b; seed=42)
    N = DR.item_count(b)
    state = env.state

    @test DR.current_epoch(state) == 1
    @test length(DR.stock(state)) == N
    @test size(DR.stock_history(state)) == (1, N)
    @test size(DR.replenishment_history(state)) == (0, N)
    @test size(DR.sales_history(state)) == (0, N)
    @test length(DR.customer_history(state)) == 0
    @test state.current_cost == 0.0
    @test DR.stock_ini(state) == DR.stock_ini(env)

    # after one step
    reward = step!(env, DR.y_oracle(env, zeros(Int, N), env.stock_ini))
    @test DR.current_epoch(state) == 2
    @test size(DR.stock_history(state)) == (2, N)
    @test size(DR.replenishment_history(state)) == (1, N)
    @test size(DR.sales_history(state)) == (1, N)
    @test length(DR.customer_history(state)) == 1
    @test state.current_cost == reward
end

@testset "DynamicReplenishment - Observe" begin
    b = DynamicReplenishmentBenchmark()
    env = DR.Environment(b; seed=42)
    N = DR.item_count(b)
    UB = DR.UB_item(b)

    x, state = observe(env)

    # x is stock_features' : (nb_features, N*UB)
    @test size(x, 2) == N * UB
    @test size(x, 1) >= DR.feature_count(b) + 1
    static_features = x[1:(DR.feature_count(b) + 1), :]
    for i in 1:N
        ref = static_features[:, (i - 1) * UB + 1]
        block = static_features[:, ((i - 1) * UB + 1):(i * UB)]
        @test all(block .≈ ref)
    end
end

@testset "DynamicReplenishment - Statistical Model" begin
    b = DynamicReplenishmentBenchmark()
    N = DR.item_count(b)
    UB = DR.UB_item(b)

    model = generate_statistical_model(b)
    @test model isa DR.statistical_model

    env = DR.Environment(b; seed=42)
    x, _ = observe(env)

    θη = model(x, N, UB)
    # θ : N outputs from θ_model, η : N*UB outputs from η_model
    @test length(θη) == N + N * UB
    @test all(isfinite.(θη))
end

@testset "DynamicReplenishment - Policies" begin
    b = DynamicReplenishmentBenchmark()
    environments = generate_environments(b, 5; seed=0)
    policies = generate_baseline_policies(b)

    @test policies.greedy.name == "Greedy"
    @test policies.random.name == "Random"

    r_greedy, _ = evaluate_policy!(policies.greedy, environments)
    @test length(r_greedy) == length(environments)
    env = environments[1]
    reset!(env)
    action = policies.greedy(env)
    @test DR.is_feasible(env.state, action)
end

@testset "DynamicReplenishment - Anticipative Solver" begin
    b = DynamicReplenishmentBenchmark(; N=5, max_steps=3)
    env = DR.Environment(b; seed=42)

    obj, trajectory = DR.anticipative_solver(env)
    policies = generate_baseline_policies(b)
    r_greedy, _ = evaluate_policy!(policies.greedy, [env])
    @test length(trajectory) == DR.max_steps(b)
    @test r_greedy[1] <= obj
    for sample in trajectory
        @test DR.is_feasible(sample.state, sample.y)
    end
    @test trajectory[end].state.current_cost == obj
end

@testset "DynamicReplenishment - Plots" begin
    using Plots

    b = DynamicReplenishmentBenchmark()
    envs = generate_environments(b, 2; seed=0)
    policies = generate_baseline_policies(b)
    _, traj = evaluate_policy!(policies.greedy, envs)

    @test has_visualization(b)
    fig1 = plot_sample(b, traj[1])
    @test fig1 isa Plots.Plot
    fig2 = plot_trajectory(b, traj)
    @test fig2 isa Plots.Plot
end