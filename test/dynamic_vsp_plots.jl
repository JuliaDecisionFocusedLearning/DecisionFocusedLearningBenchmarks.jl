@testitem "Dynamic VSP Plots" begin
    using DecisionFocusedLearningBenchmarks.DynamicVehicleScheduling
    const DVSP = DecisionFocusedLearningBenchmarks.DynamicVehicleScheduling
    using Plots

    # Create test benchmark and data (similar to scripts/a.jl)
    b = DynamicVehicleSchedulingBenchmark(; two_dimensional_features=true)
    dataset = generate_dataset(b, 3)
    environments = generate_environments(b, dataset; seed=0)
    env = environments[1]

    # Test basic plotting functions
    fig1 = DVSP.plot_instance(env)
    @test fig1 isa Plots.Plot

    instance = dataset[1].instance
    scenario = generate_scenario(b, instance; seed=0)
    v, y = generate_anticipative_solution(b, env, scenario; nb_epochs=3, reset_env=true)

    fig2 = DVSP.plot_epochs(y)
    @test fig2 isa Plots.Plot

    policies = generate_policies(b)
    lazy = policies[1]
    _, d = evaluate_policy!(lazy, env)
    fig3 = DVSP.plot_routes(d[1].info.state, d[1].y)
    @test fig3 isa Plots.Plot

    # Test animation
    temp_filename = tempname() * ".gif"
    try
        anim = DVSP.animate_epochs(y; filename=temp_filename, fps=1)
        @test anim isa Plots.AnimatedGif || anim isa Plots.Animation
        @test isfile(temp_filename)
    finally
        if isfile(temp_filename)
            rm(temp_filename)
        end
    end
end
