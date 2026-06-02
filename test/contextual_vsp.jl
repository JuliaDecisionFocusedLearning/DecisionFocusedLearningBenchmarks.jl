@testset "Contextual Stochastic VSP" begin
    using DecisionFocusedLearningBenchmarks
    using DecisionFocusedLearningBenchmarks.StochasticVehicleScheduling
    using Graphs
    using Plots
    using StableRNGs: StableRNG

    b = ContextualStochasticVehicleSchedulingBenchmark(; nb_tasks=15, nb_scenarios=4)

    @test is_exogenous(b)

    # Custom cost fields thread through to the instance
    b_custom = ContextualStochasticVehicleSchedulingBenchmark(;
        nb_tasks=10, vehicle_cost=100.0, delay_cost=5.0, p_storm=0.3, storm_multiplier=20.0
    )
    sample_custom = generate_dataset(
        b_custom, 1; contexts_per_instance=1, nb_scenarios=1, seed=99
    )[1]
    @test sample_custom.instance.vehicle_cost == 100.0
    @test sample_custom.instance.delay_cost == 5.0

    N = 2
    M = 3
    K = 2

    # N instances × M contexts × K scenarios = N*M*K unlabeled samples
    unlabeled = generate_dataset(b, N; contexts_per_instance=M, nb_scenarios=K, seed=1)
    @test length(unlabeled) == N * M * K
    @test hasproperty(unlabeled[1].extra, :scenario)
    @test unlabeled[1].extra.scenario isa VSPScenario
    @test unlabeled[1].extra.scenario.storm_active isa Bool

    # Each sample carries μ/σ and storm fields in context
    @test hasproperty(unlabeled[1].context, :district_μ)
    @test hasproperty(unlabeled[1].context, :district_σ)
    @test length(unlabeled[1].district_μ) == length(unlabeled[1].instance.city.districts)
    @test length(unlabeled[1].district_σ) == length(unlabeled[1].instance.city.districts)
    @test hasproperty(unlabeled[1].context, :storm_district)
    @test hasproperty(unlabeled[1].context, :p_storm)
    @test hasproperty(unlabeled[1].context, :storm_multiplier)
    @test unlabeled[1].storm_district isa Int
    @test 0 < unlabeled[1].p_storm < 1
    @test unlabeled[1].storm_multiplier > 1

    # storm_district is always drawn from occupied districts (≥1 task start_point inside)
    let city = unlabeled[1].instance.city
        lin = LinearIndices(city.districts)
        occupied = unique([
            lin[StochasticVehicleScheduling.get_district(t.start_point, city)...] for
            t in city.tasks[2:(end - 1)]
        ])
        @test unlabeled[1].storm_district in occupied
    end

    # Samples sharing an instance have identical city, but different μ/σ across contexts
    same_instance_block = unlabeled[1:(M * K)]
    cities = unique(s.instance.city for s in same_instance_block)
    @test length(cities) == 1
    μs = [s.district_μ for s in same_instance_block[1:K:end]]  # one per context
    @test length(unique(μs)) == M
    # storm_district is re-drawn per context; coincidences are possible but unlikely
    storm_districts = [s.storm_district for s in same_instance_block[1:K:end]]
    @test all(d -> d isa Int, storm_districts)

    # Features are 20 × ne(graph), Float32
    # Layout matches compute_features (non-contextual):
    #   1: travel time, 2: vehicle cost (source arcs), 3-11: slack deciles, 12-20: slack CDF
    sample = unlabeled[1]
    E = ne(sample.instance.graph)
    @test size(sample.x) == (20, E)
    @test eltype(sample.x) === Float32
    # Travel-time feature (row 1) must be non-negative
    @test all(sample.x[1, :] .>= 0)
    # Deciles (rows 3–11) must be non-decreasing per arc
    @test all(all(sample.x[k, :] .<= sample.x[k + 1, :]) for k in 3:10)
    # CDF features (rows 12–20) must be in [0, 1]
    @test all(0 .<= sample.x[12:20, :] .<= 1)

    # Statistical model + maximizer pipeline
    model = generate_statistical_model(b; seed=0)
    @test size(model[1].weight) == (1, 20)
    maximizer = generate_maximizer(b)
    θ = model(sample.x)
    @test length(θ) == E
    y = maximizer(θ; sample.context...)
    @test y isa BitVector
    @test length(y) == E

    # Baseline policies
    policies = generate_baseline_policies(b)
    @test hasproperty(policies, :saa)
    @test hasproperty(policies, :deterministic)
    @test hasproperty(policies, :local_search)

    # Labeled dataset via SAA policy: N*M labeled samples
    saa_dataset = generate_dataset(
        b,
        N;
        contexts_per_instance=M,
        nb_scenarios=K,
        seed=0,
        rng=StableRNG(0),
        target_policy=policies.saa,
    )
    @test length(saa_dataset) == N * M
    @test hasproperty(saa_dataset[1].extra, :scenarios)
    @test length(saa_dataset[1].extra.scenarios) == K
    @test saa_dataset[1].y isa BitVector

    # Plot extension
    @test has_visualization(b)
    figure_1 = plot_context(b, saa_dataset[1])
    @test figure_1 isa Plots.Plot
    figure_2 = plot_sample(b, saa_dataset[1])
    @test figure_2 isa Plots.Plot

    # compute_gap returns a finite number
    gap = compute_gap(b, saa_dataset, model, maximizer)
    @test isfinite(gap)

    # Objective value: ensure regenerating scenario from same context reproduces consistent value
    rng = StableRNG(7)
    fresh_scenario = generate_scenario(b, rng; sample.context...)
    @test fresh_scenario isa VSPScenario
    obj = objective_value(b, sample, y, fresh_scenario)
    @test isfinite(obj)

    # Anticipative solvers (1-scenario)
    anticipative_solver = generate_anticipative_solver(b)
    y_anti = anticipative_solver(sample.scenario; sample.context...)
    @test y_anti isa BitVector

    parametric_solver = generate_parametric_anticipative_solver(b)
    θ_zero = zeros(E)
    y_zero = parametric_solver(θ_zero, sample.scenario; sample.context...)
    @test y_zero == y_anti
end
