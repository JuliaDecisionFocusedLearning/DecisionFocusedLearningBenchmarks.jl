module StochasticVehicleScheduling

export StochasticVehicleSchedulingBenchmark
export generate_dataset, generate_maximizer, generate_statistical_model
export compact_linearized_mip,
    compact_mip, column_generation_algorithm, local_search, deterministic_mip
export evaluate_solution, is_feasible
export VSPScenario, build_stochastic_instance

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using ConstrainedShortestPaths:
    stochastic_routing_shortest_path, stochastic_routing_shortest_path_with_threshold
using Distributions: Distribution, LogNormal, Uniform, DiscreteUniform
using Flux: Chain, Dense
using Graphs:
    AbstractGraph,
    SimpleDiGraph,
    add_edge!,
    nv,
    ne,
    edges,
    src,
    dst,
    has_edge,
    inneighbors,
    outneighbors
using JuMP:
    JuMP, Model, @variable, @objective, @constraint, optimize!, value, set_silent, dual
using Printf: @printf
using Random: Random, AbstractRNG, MersenneTwister
using SparseArrays: sparse, SparseMatrixCSC
using Statistics: quantile, mean

include("utils.jl")
include("instance/constants.jl")
include("instance/task.jl")
include("instance/district.jl")
include("instance/city.jl")
include("instance/features.jl")
include("instance/instance.jl")

include("scenario.jl")

include("solution/solution.jl")
include("solution/algorithms/mip.jl")
include("solution/algorithms/column_generation.jl")
include("solution/algorithms/local_search.jl")
include("solution/algorithms/deterministic_mip.jl")
include("solution/algorithms/anticipative_solver.jl")

include("maximizer.jl")

"""
$TYPEDEF

Data structure for a stochastic vehicle scheduling benchmark.

# Fields
$TYPEDFIELDS
"""
@kwdef struct StochasticVehicleSchedulingBenchmark <: AbstractStochasticBenchmark{true}
    "number of tasks in each instance"
    nb_tasks::Int = 25
    "number of scenarios in each instance (only used to compute features and objective evaluation)"
    nb_scenarios::Int = 10
end

include("policies.jl")

function Utils.objective_value(
    ::StochasticVehicleSchedulingBenchmark,
    sample::DataSample,
    y::BitVector,
    scenario::VSPScenario,
)
    stoch = build_stochastic_instance(sample.instance, [scenario])
    return evaluate_solution(y, stoch)
end

function Utils.objective_value(
    bench::StochasticVehicleSchedulingBenchmark, sample::DataSample, y::BitVector
)
    if hasproperty(sample.extra, :scenario)
        return Utils.objective_value(bench, sample, y, sample.extra.scenario)
    elseif hasproperty(sample.extra, :scenarios)
        stoch = build_stochastic_instance(sample.instance, sample.extra.scenarios)
        return evaluate_solution(y, stoch)
    end
    return error("Sample must have scenario or scenarios")
end

"""
$TYPEDSIGNATURES

Draw a single fresh [`VSPScenario`](@ref) for the given instance.
Requires `store_city=true` (the default) when generating instances.
"""
function Utils.generate_scenario(
    ::StochasticVehicleSchedulingBenchmark, rng::AbstractRNG; instance::Instance, kwargs...
)
    @assert !isnothing(instance.city) "`generate_scenario` requires `store_city=true`"
    return draw_scenario(instance.city, instance.graph, rng)
end

"""
$TYPEDSIGNATURES
"""
function Utils.generate_baseline_policies(bench::StochasticVehicleSchedulingBenchmark)
    return svs_generate_baseline_policies(bench)
end

"""
$TYPEDSIGNATURES

Return the anticipative solver: a callable `(scenario::VSPScenario; instance, kwargs...) -> y`
that solves the 1-scenario stochastic VSP.
"""
function Utils.generate_anticipative_solver(::StochasticVehicleSchedulingBenchmark)
    return AnticipativeSolver()
end

"""
$TYPEDSIGNATURES

Return the parametric anticipative solver: a callable `(θ, scenario::VSPScenario; instance, kwargs...) -> y`.
"""
function Utils.generate_parametric_anticipative_solver(
    ::StochasticVehicleSchedulingBenchmark
)
    return AnticipativeSolver()
end

"""
$TYPEDSIGNATURES

Generate an unlabeled instance for the given `StochasticVehicleSchedulingBenchmark`.
Returns a [`DataSample`](@ref) with features `x` and `instance` set, but `y=nothing`.

To obtain labeled samples, pass a `target_policy` to [`generate_dataset`](@ref):

```julia
policy = sample -> DataSample(; sample.context..., x=sample.x,
                                y=column_generation_algorithm(sample.instance))
dataset = generate_dataset(benchmark, N; target_policy=policy)
```

If `store_city=false`, coordinates and city information are not stored in the instance,
and `generate_scenario` will not work. This can be used to save memory if you only need to evaluate
solutions on a fixed set of scenarios.
"""
function Utils.generate_instance(
    benchmark::StochasticVehicleSchedulingBenchmark,
    rng::AbstractRNG;
    store_city=true,
    kwargs...,
)
    (; nb_tasks, nb_scenarios) = benchmark
    instance = Instance(; nb_tasks, nb_scenarios, rng, store_city)
    x = get_features(instance)
    return DataSample(; x, instance)
end

"""
$TYPEDEF

Deterministic vsp maximizer for the [StochasticVehicleSchedulingBenchmark](@ref).
"""
struct StochasticVechicleSchedulingMaximizer{M}
    "mip solver model to use"
    model_builder::M
end

"""
$TYPEDSIGNATURES
"""
function Utils.generate_maximizer(
    ::StochasticVehicleSchedulingBenchmark; model_builder=highs_model
)
    return StochasticVechicleSchedulingMaximizer(model_builder)
end

"""
$TYPEDSIGNATURES

Apply the maximizer with the stored model builder.
"""
function (maximizer::StochasticVechicleSchedulingMaximizer)(
    θ::AbstractVector; instance::Instance, kwargs...
)
    return vsp_maximizer(θ; instance, model_builder=maximizer.model_builder, kwargs...)
end

"""
$TYPEDSIGNATURES
"""
function Utils.generate_statistical_model(
    ::StochasticVehicleSchedulingBenchmark; seed=nothing
)
    Random.seed!(seed)
    return Chain(Dense(20 => 1; bias=false), vec)
end

end
