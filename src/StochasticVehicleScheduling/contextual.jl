"""
$TYPEDEF

Contextual variant of the stochastic vehicle scheduling benchmark.

The instance fixes the city layout, tasks, and inter-area distribution; only the
per-district `(μ_d, σ_d)` parameters are re-drawn per context (from
`default_district_μ` and `default_district_σ`) and exposed to the predictive model as
features. Scenarios are then sampled fresh per context using its `(μ_d, σ_d)`.

# Fields
$TYPEDFIELDS
"""
@kwdef struct ContextualStochasticVehicleSchedulingBenchmark <:
              AbstractStochasticBenchmark{true}
    "number of tasks in each instance"
    nb_tasks::Int = 25
    "number of scenarios drawn per context (used for objective evaluation)"
    nb_scenarios::Int = 10
end

"""
$TYPEDSIGNATURES

Build a `City` for the contextual stochastic VSP variant.

Differs from [`create_random_city`](@ref) in three ways:
- `nb_scenarios=0`: no internal scenario storage (per-context scenarios are drawn on demand
   by [`generate_scenario`](@ref) for the contextual benchmark).
- Per-district `random_delay` is a placeholder (`LogNormal(0.0, 1.0)`): the contextual
   scenario generator builds `LogNormal`s from the sample's `(district_μ, district_σ)`
   instead of reading the city's stored ones.
- No `generate_scenarios!` / `compute_perturbed_end_times!` pass (nothing to roll into).
"""
function create_contextual_city(;
    nb_tasks::Int=default_nb_tasks,
    rng::AbstractRNG=Random.default_rng(),
    seed=nothing,
    αᵥ_low=default_αᵥ_low,
    αᵥ_high=default_αᵥ_high,
    first_begin_time=default_first_begin_time,
    last_begin_time=default_last_begin_time,
    task_μ=default_task_μ,
    task_σ=default_task_σ,
    city_kwargs...,
)
    isnothing(seed) || Random.seed!(rng, seed)
    city = City(; nb_scenarios=0, nb_tasks, city_kwargs...)
    placeholder = LogNormal(0.0, 1.0)
    nb_per_edge = city.width ÷ city.district_width
    for x in 1:nb_per_edge, y in 1:nb_per_edge
        city.districts[x, y] = District(; random_delay=placeholder, nb_scenarios=0)
    end
    init_tasks!(
        city, αᵥ_low, αᵥ_high, first_begin_time, last_begin_time, task_μ, task_σ; rng=rng
    )
    return city
end

"""
$TYPEDSIGNATURES

Generate an instance for the contextual stochastic VSP benchmark.

The returned `DataSample` carries the city + graph in `context.instance` and leaves
`x=nothing`; per-district `(μ_d, σ_d)` and the feature matrix are added later by
[`generate_context`](@ref).
"""
function Utils.generate_instance(
    benchmark::ContextualStochasticVehicleSchedulingBenchmark, rng::AbstractRNG; kwargs...
)
    (; nb_tasks) = benchmark
    city = create_contextual_city(; nb_tasks, rng, kwargs...)
    graph = create_VSP_graph(city)
    instance = Instance(
        graph,
        zeros(0, ne(graph)),
        zeros(0, 0),
        zeros(0, 0),
        city.vehicle_cost,
        city.delay_cost,
        city,
    )
    return DataSample(; instance)
end

"""
$TYPEDSIGNATURES

Build the `5 × ne(graph)` per-edge feature matrix for the contextual stochastic VSP.

Each column `e = (u, v)` is `[μ_src, σ_src, μ_dst, σ_dst, travel_time]`, where the
source district is the district containing `tasks[u].end_point` and the destination
district is the district containing `tasks[v].start_point` (matching the slack
computation in `draw_scenario`). `district_μ` and `district_σ` are indexed via
`LinearIndices(city.districts)`.
"""
function compute_contextual_features(
    city::City, graph::AbstractGraph, district_μ::AbstractVector, district_σ::AbstractVector
)
    lin = LinearIndices(city.districts)
    features = Matrix{Float32}(undef, 5, ne(graph))
    for (i, e) in enumerate(edges(graph))
        u, v = src(e), dst(e)
        ox, oy = get_district(city.tasks[u].end_point, city)
        dx, dy = get_district(city.tasks[v].start_point, city)
        o = lin[ox, oy]
        d = lin[dx, dy]
        travel_time = distance(city.tasks[u].end_point, city.tasks[v].start_point)
        features[1, i] = district_μ[o]
        features[2, i] = district_σ[o]
        features[3, i] = district_μ[d]
        features[4, i] = district_σ[d]
        features[5, i] = travel_time
    end
    return features
end

"""
$TYPEDSIGNATURES

Draw `(district_μ, district_σ)` for every district from `default_district_μ` and
`default_district_σ`, add them to `instance_sample.context`, and compute the per-edge
feature matrix `x`.
"""
function Utils.generate_context(
    ::ContextualStochasticVehicleSchedulingBenchmark,
    rng::AbstractRNG,
    instance_sample::DataSample,
)
    instance = instance_sample.context.instance
    city = instance.city
    nb_districts = length(city.districts)
    district_μ = rand(rng, default_district_μ, nb_districts)
    district_σ = rand(rng, default_district_σ, nb_districts)
    x = compute_contextual_features(city, instance.graph, district_μ, district_σ)
    return DataSample(;
        x, instance_sample.context..., district_μ=district_μ, district_σ=district_σ
    )
end

"""
$TYPEDSIGNATURES

Draw a fresh [`VSPScenario`](@ref) using the context's `(district_μ, district_σ)`
to build per-district `LogNormal` distributions on the fly. Solver kwargs `instance`,
`district_μ`, `district_σ` are spread from `sample.context`.
"""
function Utils.generate_scenario(
    ::ContextualStochasticVehicleSchedulingBenchmark,
    rng::AbstractRNG;
    instance::Instance,
    district_μ::AbstractVector,
    district_σ::AbstractVector,
    kwargs...,
)
    city = instance.city
    @assert !isnothing(city) "contextual SVS `generate_scenario` requires `store_city=true`"
    lin = LinearIndices(city.districts)
    district_delay_fn = (x, y) -> begin
        i = lin[x, y]
        return LogNormal(district_μ[i], district_σ[i])
    end
    return draw_scenario(city, instance.graph, rng; district_delay_fn)
end

"""
$TYPEDSIGNATURES

Small MLP mapping the `5×ne(graph)` per-edge feature matrix to a length-`ne(graph)`
score vector. Architecture: `Chain(Dense(5 => 16, relu), Dense(16 => 1; bias=false), vec)`.
"""
function Utils.generate_statistical_model(
    ::ContextualStochasticVehicleSchedulingBenchmark; seed=nothing
)
    Random.seed!(seed)
    return Chain(Dense(5 => 1; bias=false), vec)
end

"""
$TYPEDSIGNATURES

Return the same deterministic VSP maximizer as the non-contextual benchmark.
"""
function Utils.generate_maximizer(
    ::ContextualStochasticVehicleSchedulingBenchmark; model_builder=highs_model
)
    return StochasticVechicleSchedulingMaximizer(model_builder)
end

"""
$TYPEDSIGNATURES

Return the anticipative solver: a callable `(scenario::VSPScenario; instance, kwargs...) -> y`
that solves the 1-scenario stochastic VSP. Identical to the non-contextual variant;
per-context `(district_μ, district_σ)` only affect scenario sampling, not the oracle.

# Keyword Arguments
- `model_builder`: a function returning an empty `JuMP.Model` with a solver attached (defaults to `scip_model`).
"""
function Utils.generate_anticipative_solver(
    ::ContextualStochasticVehicleSchedulingBenchmark; model_builder=scip_model
)
    return AnticipativeSolver(; model_builder=model_builder)
end

"""
$TYPEDSIGNATURES

Return the parametric anticipative solver: a callable
`(θ, scenario::VSPScenario; instance, kwargs...) -> y` that solves the parametric
anticipative subproblem `argmin_y c(y, scenario) + θᵀy`.

# Keyword Arguments
- `model_builder`: a function returning an empty `JuMP.Model` with a solver attached (defaults to `scip_model`).
"""
function Utils.generate_parametric_anticipative_solver(
    ::ContextualStochasticVehicleSchedulingBenchmark; model_builder=scip_model
)
    return AnticipativeSolver(; model_builder=model_builder)
end

"""
$TYPEDSIGNATURES

Forward to the SVS baseline policies. These operate on `sample.instance` and scenarios,
so they work unchanged for the contextual variant.
"""
function Utils.generate_baseline_policies(::ContextualStochasticVehicleSchedulingBenchmark)
    return (;
        deterministic=Policy(
            "Deterministic MIP", "Ignores delays", svs_deterministic_policy
        ),
        saa=Policy("SAA (col gen)", "Stochastic MIP over K scenarios", svs_saa_policy),
        saa_mip=Policy(
            "SAA (exact MIP)",
            "Exact stochastic MIP over K scenarios via compact linearized formulation",
            svs_saa_mip_policy,
        ),
        local_search=Policy(
            "Local search", "Heuristic with K scenarios", svs_local_search_policy
        ),
    )
end

function Utils.objective_value(
    ::ContextualStochasticVehicleSchedulingBenchmark,
    sample::DataSample,
    y::BitVector,
    scenario::VSPScenario,
)
    stoch = build_stochastic_instance(sample.instance, [scenario])
    return evaluate_solution(y, stoch)
end

function Utils.objective_value(
    bench::ContextualStochasticVehicleSchedulingBenchmark, sample::DataSample, y::BitVector
)
    if hasproperty(sample.extra, :scenario)
        return Utils.objective_value(bench, sample, y, sample.extra.scenario)
    elseif hasproperty(sample.extra, :scenarios)
        stoch = build_stochastic_instance(sample.instance, sample.extra.scenarios)
        return evaluate_solution(y, stoch)
    end
    return error("Sample must have scenario or scenarios")
end
