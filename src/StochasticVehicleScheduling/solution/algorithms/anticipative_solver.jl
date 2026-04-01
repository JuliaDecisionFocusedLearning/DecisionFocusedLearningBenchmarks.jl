@kwdef struct AnticipativeSolver{M,A}
    model_builder::M = scip_model
    single_scenario_algorithm::A = compact_mip
end

function Base.show(io::IO, ::AnticipativeSolver)
    return print(io, "Anticipative solver for StochasticVehicleSchedulingBenchmark")
end

function (solver::AnticipativeSolver)(scenario; instance::Instance, kwargs...)
    stochastic_inst = build_stochastic_instance(instance, [scenario])
    return solver.single_scenario_algorithm(
        stochastic_inst; model_builder=solver.model_builder, kwargs...
    )
end

function (solver::AnticipativeSolver)(θ, scenario; instance::Instance, kwargs...)
    stochastic_inst = build_stochastic_instance(instance, [scenario])
    return solver.single_scenario_algorithm(
        stochastic_inst, θ; model_builder=solver.model_builder, kwargs...
    )
end
