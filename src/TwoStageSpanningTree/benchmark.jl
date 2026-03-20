"""
$TYPEDEF

Two-Stage Minimum Spanning Tree benchmark.

An instance consists of a grid graph of size `n × m` with:
- Random first-stage costs for each edge (drawn from `c_range`), known before the scenario.
- `nb_scenarios` second-stage cost draws (from `d_range`) embedded in the instance and used
  for computing features via [`compute_features`](@ref).

A fresh evaluation scenario is drawn independently by [`generate_scenario`](@ref)
and is used for labeling and objective evaluation.

# Fields
$TYPEDFIELDS
"""
@kwdef struct TwoStageSpanningTreeBenchmark <: Utils.AbstractStochasticBenchmark{true}
    "Number of grid rows."
    n::Int = 4
    "Number of grid columns."
    m::Int = 4
    "Number of second-stage cost scenarios embedded in the instance (for features)."
    nb_scenarios::Int = 10
    "Range for first-stage edge costs."
    c_range::UnitRange{Int} = 1:10
    "Range for second-stage edge costs."
    d_range::UnitRange{Int} = 1:20
end

function Utils.generate_instance(
    bench::TwoStageSpanningTreeBenchmark, rng::AbstractRNG; kwargs...
)
    (; n, m, nb_scenarios, c_range, d_range) = bench
    g = Graphs.grid((n, m))
    E = ne(g)
    c = Float32[rand(rng, c_range) for _ in 1:E]
    d = Float32[rand(rng, d_range) for _ in 1:E, _ in 1:nb_scenarios]
    instance = TwoStageSpanningTreeInstance(g, c, d)
    x = compute_features(instance; c_max=maximum(c_range), d_max=maximum(d_range))
    return Utils.DataSample(; x, instance)
end

function Utils.generate_scenario(
    bench::TwoStageSpanningTreeBenchmark,
    rng::AbstractRNG;
    instance::TwoStageSpanningTreeInstance,
    kwargs...,
)
    return Float32[rand(rng, bench.d_range) for _ in 1:ne(instance.graph)]
end

function Utils.generate_statistical_model(::TwoStageSpanningTreeBenchmark; seed=nothing)
    isnothing(seed) || Random.seed!(seed)
    nb_features = 2 + 7 * length(0.0:0.1:1.0)
    return Flux.Chain(Flux.Dense(nb_features => 1; bias=false), vec)
end

struct TSTMaximizer end

function (::TSTMaximizer)(
    θ::AbstractVector; instance::TwoStageSpanningTreeInstance, kwargs...
)
    return maximum_weight_forest(instance.graph, θ)
end

Utils.generate_maximizer(::TwoStageSpanningTreeBenchmark) = TSTMaximizer()

Utils.is_minimization_problem(::TwoStageSpanningTreeBenchmark) = true

function _complete_second_stage(
    y::AbstractVector, graph::AbstractGraph, scenario::AbstractVector
)
    weights = copy(scenario)
    m = min(zero(eltype(weights)), minimum(weights) - one(eltype(weights)))
    weights[y .> 0] .= m
    _, tree = kruskal(graph, weights)
    return tree .⊻ BitVector(y .> 0)
end

function Utils.objective_value(
    ::TwoStageSpanningTreeBenchmark, sample::Utils.DataSample, y::AbstractVector
)
    (; instance) = sample.context
    scenarios = if hasproperty(sample.extra, :scenarios)
        sample.extra.scenarios
    else
        [instance.second_stage_costs[:, s] for s in 1:nb_scenarios(instance)]
    end
    return dot(y, instance.first_stage_costs) +
           mean(dot(_complete_second_stage(y, instance.graph, ξ), ξ) for ξ in scenarios)
end

function Utils.generate_anticipative_solver(::TwoStageSpanningTreeBenchmark)
    return (scenario::AbstractVector; instance::TwoStageSpanningTreeInstance, kwargs...) ->
        begin
            c, g = instance.first_stage_costs, instance.graph
            _, tree = kruskal(g, min.(c, scenario))
            return BitVector(tree .& (c .<= scenario))
        end
end

function Utils.generate_baseline_policies(::TwoStageSpanningTreeBenchmark)
    function make_saa_policy(solver_fn)
        return (sample, scenarios) -> begin
            d_eval = reduce(hcat, scenarios)
            saa_inst = TwoStageSpanningTreeInstance(
                sample.instance.graph, sample.instance.first_stage_costs, d_eval
            )
            sol = solver_fn(saa_inst)
            return [
                Utils.DataSample(;
                    sample.context...,
                    x=sample.x,
                    y=sol.y,
                    extra=(; scenarios=scenarios),
                ),
            ]
        end
    end
    return (;
        cut_generation=make_saa_policy(inst -> cut_generation(inst; verbose=false)),
        benders=make_saa_policy(inst -> benders_decomposition(inst; verbose=false)),
        column_heuristic=make_saa_policy(inst -> column_heuristic(inst; verbose=false)),
        lagrangian=make_saa_policy(inst -> first(lagrangian_relaxation(inst))),
    )
end
