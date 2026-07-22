const DR = DecisionFocusedLearningBenchmarks.DynamicReplenishment

@testset "DynamicReplenishment - Benchmark Construction" begin
    b = DynamicReplenishmentBenchmark()
    @test b.N == 10
    @test b.λ == 15
    @test b.d == 5
    @test b.stock_inf == 0
    @test b.stock_sup == 30
    @test b.ub_same_item == 30
    @test b.delivery_delay == 3
    @test b.max_steps == 10
    @test size(b.constraints_matrix) == (12, 10)
    @test size(b.quotas) == (10, 12)
    @test size(b.max_quotas) == (b.max_steps, b.N)
    @test all(b.max_quotas .≥ 0)
    @test all(b.max_quotas .≤ b.ub_same_item)

    b_custom = DynamicReplenishmentBenchmark(;
        N=5,
        λ=10,
        constraints_matrix=[1 1 1 0 0; 0 0 0 1 1; 0 0 0 0 1],
        quotas=[20 15 5; 10 20 5],
        d=3,
        stock_inf=2,
        stock_sup=10,
        ub_same_item=17,
        delivery_delay=1,
        max_steps=2,
    )
    @test b_custom.N == 5
    @test b_custom.λ == 10
    @test b_custom.d == 3
    @test b_custom.stock_inf == 2
    @test b_custom.stock_sup == 10
    @test b_custom.ub_same_item == 17
    @test b_custom.delivery_delay == 1
    @test b_custom.max_steps == 2
    @test size(b_custom.constraints_matrix) == (8, 5)
    @test b_custom.quotas[1, :] == [20, 15, 5, 17, 17, 17, 17, 17]
    @test size(b_custom.quotas) == (2, 8)

    @test b_custom.max_quotas[1, :] == [17, 17, 17, 15, 5]
    @test b_custom.max_quotas[2, :] == [10, 10, 10, 17, 5]

    @test DR.item_count(b) == 10
    @test DR.feature_count(b) == 5
    @test DR.max_steps(b) == 10
    @test DR.stock_inf(b) == 0
    @test DR.stock_sup(b) == 30
    @test DR.ub_same_item(b) == 30
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
    rng = Xoshiro(42)
    env1 = DR.Environment(b, rng)
    @test !is_terminated(env1)
    @test DR.item_count(env1) == 10
    @test DR.max_steps(env1) == 10
    @test length(DR.stock_ini(env1)) == 10
    @test all(0 .≤ DR.stock_ini(env1) .≤ 5)

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
    env2 = DR.Environment(b, rng; stock_ini=fill(5, 10))
    @test DR.stock_ini(env2) == fill(5, 10)
    @test DR.stock(env2) == fill(5, 10)
end

@testset "DynamicReplenishment - Environment Reset" begin
    b = DynamicReplenishmentBenchmark()
    rng = Xoshiro(42)
    env = DR.Environment(b, rng)

    s0 = copy(DR.stock_ini(env))
    N = DR.item_count(b)
    repl = zeros(Int, N)
    step!(env, repl, rng)
    reset!(env, rng)

    @test !is_terminated(env)
    @test DR.stock(env) == s0
    @test DR.current_epoch(env) == 1
end

@testset "DynamicReplenishment - Environment Step" begin
    b = DynamicReplenishmentBenchmark()
    rng = Xoshiro(42)
    env = DR.Environment(b, rng)
    N = DR.item_count(b)

    action = zeros(Int, N)
    reward = step!(env, action, rng)
    @test reward isa Float64
    @test DR.current_epoch(env) == 2

    # run to termination
    while !is_terminated(env)
        repl = zeros(Int, N)
        @test DR.is_feasible(env.state, repl)
        step!(env, repl, rng)
    end
    @test is_terminated(env)
    @test_throws AssertionError step!(env, zeros(Int, N), rng)
end

@testset "DynamicReplenishment - Feasibility" begin
    b = DynamicReplenishmentBenchmark(
        N=2, max_steps=2, constraints_matrix=[1 1], quotas=[1; 1]
    )
    N = DR.item_count(b)
    stock_ini = [0, 0]
    rng = Xoshiro(42)
    env = DR.Environment(b, rng; stock_ini=stock_ini)
    # zero replenishment is always feasible
    @test DR.is_feasible(env.state, zeros(Int, N))
    @test DR.is_feasible(env.state, [0, 1])
    @test DR.is_feasible(env.state, [1, 0])
    @test DR.is_feasible(env.state, [0, 0])

    # replenishment exceeding quota is infeasible
    @test !DR.is_feasible(env.state, [2, 0])
    @test !DR.is_feasible(env.state, [0, 2])
    @test !DR.is_feasible(env.state, [1, 1])
end

@testset "DynamicReplenishment - State" begin
    b = DynamicReplenishmentBenchmark()
    rng = Xoshiro(42)
    env = DR.Environment(b, rng)
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
    reward = step!(env, zeros(Int, N), rng)
    @test DR.current_epoch(state) == 2
    @test size(DR.stock_history(state)) == (2, N)
    @test size(DR.replenishment_history(state)) == (1, N)
    @test size(DR.sales_history(state)) == (1, N)
    @test length(DR.customer_history(state)) == 1
    @test state.current_cost == reward
end

@testset "DynamicReplenishment - Observe" begin
    b = DynamicReplenishmentBenchmark()
    rng = Xoshiro(42)
    env = DR.Environment(b, rng)
    N = DR.item_count(b)
    ub = DR.ub_per_item(env)

    x, state = observe(env)

    @test !any(isnan.(x))

    # x is stock_features' : (nb_features, sum(UB))
    @test size(x, 2) == sum(ub)
    @test size(x, 1) >= DR.feature_count(b) + 1
    static_features = x[1:(DR.feature_count(b) + 1), :]

    starts = [1; cumsum(ub)[1:(end - 1)] .+ 1]
    ends = cumsum(ub)
    for i in 1:N
        rows = starts[i]:ends[i]
        ref = static_features[:, starts[i]]
        block = static_features[:, rows]
        @test all(block .≈ ref)
    end
end

@testset "DynamicReplenishment - Statistical Model" begin
    b = DynamicReplenishmentBenchmark()
    N = DR.item_count(b)

    model = generate_statistical_model(b)
    @test model isa DR.statistical_model

    rng = Xoshiro(42)
    env = DR.Environment(b, rng)
    x, _ = observe(env)
    @test !any(isnan.(x))

    ub = DR.ub_per_item(env)
    θη = model(x)
    @test length(θη) == N + sum(ub)
    @test all(isfinite.(θη))
end

@testset "DynamicReplenishment - Anticipative trajectory is feasible for the CO layer" begin
    b = DynamicReplenishmentBenchmark(; N=5, max_steps=3)
    rng = Xoshiro(0)
    env = generate_environments(b, 1; seed=0)
    ant_solver = generate_anticipative_solver(b)
    ant_traj = ant_solver(env[1])
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)
    while !is_terminated(env[1])
        x, _ = observe(env[1])
        θ = model(x)
        y_true = ant_traj[DR.current_epoch(env[1].env)].y
        y_hat = maximizer(θ; state=env[1].env.state, y_true=y_true)
        @test y_true == y_hat
        step!(env[1].env, y_true, rng)
    end
end

@testset "DynamicReplenishment - Parametric Anticipative trajectory is feasible for the CO layer" begin
    b = DynamicReplenishmentBenchmark(; N=5, max_steps=3)
    rng = Xoshiro(0)
    env = generate_environments(b, 1; seed=0)
    model = generate_statistical_model(b)
    param_ant_solver = DR.generate_parametric_anticipative_solver(b)
    while !is_terminated(env[1])
        x, _ = observe(env[1])
        @test !any(isnan.(x))
        θ = model(x)
        y_true = param_ant_solver(θ, env[1].env.scenario, env[1])[1].y
        y_hat = DR.generate_maximizer(b)(θ; state=env[1].env.state, y_true=y_true)
        @test y_true == y_hat
        step!(env[1].env, y_true, rng)
    end
end
@testset "DynamicReplenishment - Policies" begin
    b = DynamicReplenishmentBenchmark()
    environments = generate_environments(b, 5; seed=0)
    policies = generate_baseline_policies(b)

    @test policies.greedy.name == "Greedy"
    @test policies.random.name == "Random"
    @test policies.lazy.name == "Lazy"

    r_greedy, _ = evaluate_policy!(policies.greedy, environments, 5)
    @test length(r_greedy) == length(environments)
    env = environments[1]
    reset!(env)
    greedy_action = policies.greedy(env.env)
    @test DR.is_feasible(env.env.state, greedy_action)
    random_action = policies.random(env.env)
    @test DR.is_feasible(env.env.state, random_action)
    lazy_action = policies.lazy(env.env)
    @test DR.is_feasible(env.env.state, lazy_action)
    @test all(lazy_action .== 0)
end

@testset "DynamicReplenishment - Anticipative Solver" begin
    b = DynamicReplenishmentBenchmark(; N=5, max_steps=3)
    rng = Xoshiro(42)
    env = generate_environments(b, 1; seed=0)
    scenario = DR.generate_scenario(b; rng=rng)
    env[1].env.scenario = scenario
    ant_obj, ant_traj = DR.anticipative_solver(env[1].env, rng, scenario)
    policies = generate_baseline_policies(b)
    r_greedy, greedy_traj = evaluate_policy!(policies.greedy, env)
    @test length(ant_traj) == DR.max_steps(b) == length(greedy_traj)
    for g_sample in greedy_traj
        @test DR.is_feasible(g_sample.instance, g_sample.y)
    end
    for ant_sample in ant_traj
        @test DR.is_feasible(ant_sample.state, ant_sample.y)
    end
    @test r_greedy[1] <= ant_obj

    # @test isapprox(ant_traj[end].state.current_cost, ant_obj; rtol=1e-5)
end

@testset "DynamicReplenishment - Parametric Anticipative Solver" begin
    b = DynamicReplenishmentBenchmark(; N=5, max_steps=3)
    rng = Xoshiro(42)
    env = generate_environments(b, 1; seed=0)
    model = generate_statistical_model(b)
    x, _ = observe(env[1])
    θ = model(x)
    param_ant_solver = DR.generate_parametric_anticipative_solver(b)
    ant_traj = param_ant_solver(θ, env[1].env.scenario, env[1])
    policies = generate_baseline_policies(b)
    r_greedy, greedy_traj = evaluate_policy!(policies.greedy, env)
    @test length(ant_traj) == DR.max_steps(b) == length(greedy_traj)
    for g_sample in greedy_traj
        @test DR.is_feasible(g_sample.instance, g_sample.y)
    end
    for ant_sample in ant_traj
        @test DR.is_feasible(ant_sample.state, ant_sample.y)
    end
end

@testset "DynamicReplenishment - Θ ̇g(y) = obj(co_layer)" begin
    b = DynamicReplenishmentBenchmark(; N=5, max_steps=5)
    rng = Xoshiro(42)
    env = generate_environments(b, 1)
    policies = generate_baseline_policies(b)
    _, traj = evaluate_policy!(policies.greedy, env)
    maximizer = generate_maximizer(b)
    model = generate_statistical_model(b)
    x_1, state_1 = traj[1].x, traj[1].instance
    N = DR.item_count(b)
    ub = DR.ub_per_item(state_1)

    Θ = model(x_1)
    Y_oracle = maximizer(Θ; state=state_1)
    Z_oracle = DR.get_z_from_y(Y_oracle, state_1)
    maximizer_obj = DR._obj_function(N, ub, Θ, Y_oracle, Z_oracle)
    Θ_dot_g_y = DR.dot(Θ, DR.g(Y_oracle; state=state_1))
    @test isapprox(maximizer_obj, Θ_dot_g_y; rtol=1e-5)
end

@testset "DynamicReplenishment - Plots" begin
    using Plots

    b = DynamicReplenishmentBenchmark()
    envs = generate_environments(b, 5)
    policies = generate_baseline_policies(b)
    _, traj = evaluate_policy!(policies.greedy, envs)

    @test has_visualization(b)
    fig1 = plot_sample(b, traj[1])
    @test fig1 isa Plots.Plot
    fig2 = plot_trajectory(b, traj)
    @test fig2 isa Plots.Plot
end
