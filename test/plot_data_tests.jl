using Test
using DecisionFocusedLearningBenchmarks
using StaticArrays

# Simple helper to build a small instance with 3 locations (depot + 2 customers)
function make_simple_instance()
    coords = [Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0)]
    service_time = [0.0, 1.0, 1.5]
    start_time = [0.0, 10.0, 12.0]
    duration = [0.0 1.0 1.5; 1.0 0.0 2.0; 1.5 2.0 0.0]
    return StaticInstance(;
        coordinate=coords,
        service_time=service_time,
        start_time=start_time,
        duration=duration,
    )
end

@testset "plot_data builder and wiring" begin
    inst = make_simple_instance()
    state = DVSPState()
    reset_state!(
        state,
        Instance(; epoch_duration=1.0, Δ_dispatch=0.0, static_instance=inst);
        indices=[2, 3],
        service_time=[1.0, 1.5],
        start_time=[10.0, 12.0],
    )

    ds = [DataSample(; instance=state, y_true=Vector{Int}[])]

    pd = build_plot_data(ds)
    @test pd.n_epochs == 1
    @test length(pd.coordinates) == 1
    @test length(pd.routes) == 1

    # Ensure plotting functions accept PlotData (smoke)
    fig = plot_epochs(pd)
    @test !isnothing(fig)

    anim = animate_epochs(pd; filename="test_anim.gif", fps=1)
    @test !isnothing(anim)
end
