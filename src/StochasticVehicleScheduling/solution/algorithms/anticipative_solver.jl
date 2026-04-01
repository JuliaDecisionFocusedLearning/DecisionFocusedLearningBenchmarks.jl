@kwdef struct AnticipativeSolver{A}
    single_scenario_algorithm::A = compact_mip
end

function Base.show(io::IO, ::AnticipativeSolver)
    return print(io, "Anticipative solver for StochasticVehicleSchedulingBenchmark")
end

function (solver::AnticipativeSolver)(scenario; instance::Instance, kwargs...)
    stochastic_inst = build_stochastic_instance(instance, [scenario])
    return solver.single_scenario_algorithm(stochastic_inst)
end

function (solver::AnticipativeSolver)(θ, scenario; instance::Instance, kwargs...)
    stochastic_inst = build_stochastic_instance(instance, [scenario])
    return solver.single_scenario_algorithm(stochastic_inst, θ)
end
