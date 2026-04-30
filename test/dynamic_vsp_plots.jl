@testset "Dynamic VSP Plots" begin
    using Plots

    # Create test benchmark and environments
    b = DynamicVehicleSchedulingBenchmark(; two_dimensional_features=true)
    environments = generate_environments(b, 3; seed=0)
    env = environments[1]

    # Get a trajectory via the anticipative solver
    y = generate_anticipative_solver(b)(env; nb_epochs=3)

    # Test plot_instance (shows first epoch state)
    fig1 = plot_instance(b, y[1])
    @test fig1 isa Plots.Plot

    # Test plot_trajectory (grid of epoch subplots)
    fig2 = plot_trajectory(b, y)
    @test fig2 isa Plots.Plot

    # Test plot_sample via baseline policy
    policies = generate_baseline_policies(b)
    lazy = policies[1]
    _, d = evaluate_policy!(lazy, env)
    fig3 = plot_sample(b, d[1])
    @test fig3 isa Plots.Plot

    # Test animate_trajectory — returns Animation, save separately with gif()
    temp_filename = tempname() * ".gif"
    try
        anim = animate_trajectory(b, y; fps=1)
        @test anim isa Plots.Animation
        gif(anim, temp_filename; fps=1)
        @test isfile(temp_filename)
    finally
        if isfile(temp_filename)
            rm(temp_filename)
        end
    end
end
