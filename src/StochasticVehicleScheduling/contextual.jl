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
    "cost of one vehicle (overrides `default_vehicle_cost` for this benchmark only)"
    vehicle_cost::Float64 = 200.0
    "cost of one minute of delay (overrides `default_delay_cost` for this benchmark only)"
    delay_cost::Float64 = 20.0
    "probability that the storm is active in each scenario"
    p_storm::Float64 = 0.15
    "delay multiplier for the storm hotspot district when the storm is active"
    storm_multiplier::Float64 = 50.0
    "number of Monte Carlo draws used to compute per-arc slack quantile features"
    nb_feat_scenarios::Int = 50
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
    (; nb_tasks, vehicle_cost, delay_cost) = benchmark
    city = create_contextual_city(; nb_tasks, rng, vehicle_cost, delay_cost, kwargs...)
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

Build the `20 × ne(graph)` per-edge feature matrix for the contextual stochastic VSP
using Monte Carlo draws from the context-defined distributions.

Mirrors [`compute_features`](@ref) from the non-contextual benchmark exactly:
- Feature 1: nominal travel time (Euclidean distance between task endpoints).
- Feature 2: vehicle cost if the arc leaves the depot source node, else 0.
- Features 3–11: deciles (0.1 … 0.9) of the empirical slack distribution over
  `nb_feat_scenarios` draws.
- Features 12–20: empirical CDF of the slack at thresholds
  `[-100, -50, -20, -10, 0, 10, 50, 200, 500]`.

The slack for arc (u→v) in one draw uses **Formula A** (identical to `compute_slacks`
in the non-contextual pipeline):
```
slack = scenario_start_time[v] - (scenario_end_time[u] + nominal_travel_time)
```
where `scenario_end_time[u]` and `scenario_start_time[v]` are drawn by replicating the
`generate_scenarios!` + `compute_perturbed_end_times!` logic with the contextual
`(district_μ, district_σ)`.  The storm is activated once per scenario draw (Bernoulli
with probability `p_storm`) and shifts `district_μ[storm_district]` by
`log(storm_multiplier)`, exactly as in [`generate_scenario`](@ref).

A forked copy of `rng` is used so the main RNG state (used afterwards for scenario
generation) is not consumed.
"""
function compute_contextual_slack_features(
    city::City,
    graph::AbstractGraph,
    district_μ::AbstractVector,
    district_σ::AbstractVector,
    storm_district::Int,
    p_storm::Float64,
    storm_multiplier::Float64,
    rng::AbstractRNG;
    nb_feat_scenarios::Int=50,
)
    lin = LinearIndices(city.districts)
    tasks = city.tasks
    N = length(tasks)
    nb_per_edge = size(city.districts, 1)
    E = ne(graph)

    cumul = [-100, -50, -20, -10, 0, 10, 50, 200, 500]
    nb_features = 2 + 9 + length(cumul)
    features = zeros(Float32, nb_features, E)

    # Use a forked RNG so the caller's state is untouched.
    rng_feat = copy(rng)

    # Accumulate per-arc slack samples across nb_feat_scenarios draws.
    slack_samples = [Vector{Float64}(undef, nb_feat_scenarios) for _ in 1:E]

    for s in 1:nb_feat_scenarios
        # --- Storm activation (once per scenario, same as generate_scenario) ---
        storm_active = rand(rng_feat) < p_storm

        # --- Inter-area factor for 24 hours (mirrors generate_scenarios!) ---
        inter_area = zeros(24)
        prev_ia = 0.0
        for h in 1:24
            prev_ia = (prev_ia + 0.1) * rand(rng_feat, city.random_inter_area_factor)
            inter_area[h] = prev_ia
        end

        # --- District delays for 24 hours (mirrors roll(district, rng)) ---
        # Uses contextual LogNormal(effective_μ, σ) with storm correction.
        district_delays = [zeros(24) for _ in 1:nb_per_edge, _ in 1:nb_per_edge]
        for x in 1:nb_per_edge
            for y in 1:nb_per_edge
                i = lin[x, y]
                effective_μ = if (storm_active && i == storm_district)
                    district_μ[i] + log(storm_multiplier)
                else
                    district_μ[i]
                end
                d = LogNormal(effective_μ, district_σ[i])
                prev_d = 0.0
                for h in 1:24
                    prev_d = prev_d / 2.0 + rand(rng_feat, d)
                    district_delays[x, y][h] = prev_d
                end
            end
        end

        # --- Task start times (mirrors roll(task, rng)) ---
        scenario_start_time = [t.start_time + rand(rng_feat, t.random_delay) for t in tasks]

        # --- Task end times (mirrors compute_perturbed_end_times!) ---
        scenario_end_time = [t.end_time for t in tasks]
        for i in 2:(N - 1)
            task = tasks[i]
            origin_x, origin_y = get_district(task.start_point, city)
            dest_x, dest_y = get_district(task.end_point, city)
            ξ₁ = scenario_start_time[i]
            ξ₂ = ξ₁ + district_delays[origin_x, origin_y][hour_of(ξ₁)]
            ξ₃ = ξ₂ + (task.end_time - task.start_time) + inter_area[hour_of(ξ₂)]
            scenario_end_time[i] = ξ₃ + district_delays[dest_x, dest_y][hour_of(ξ₃)]
        end

        # --- Slack per arc: Formula A (mirrors compute_slacks(city, u, v)) ---
        for (j, e) in enumerate(edges(graph))
            u, v = src(e), dst(e)
            travel_time = distance(tasks[u].end_point, tasks[v].start_point)
            slack_samples[j][s] =
                scenario_start_time[v] - (scenario_end_time[u] + travel_time)
        end
    end

    # --- Aggregate into feature matrix (mirrors compute_features loop) ---
    for (i, e) in enumerate(edges(graph))
        u, v = src(e), dst(e)
        features[1, i] = distance(tasks[u].end_point, tasks[v].start_point)
        features[2, i] = u == 1 ? Float32(city.vehicle_cost) : 0.0f0
        slacks = slack_samples[i]
        features[3:11, i] = quantile(slacks, [0.1 * k for k in 1:9])
        features[12:nb_features, i] = [mean(slacks .<= x) for x in cumul]
    end

    return features
end

"""
$TYPEDSIGNATURES

Compute the per-feature standard deviation across all edges in `dataset`.

Concatenates the `7 × ne` feature matrices of all samples horizontally and returns a
length-7 vector of row-wise standard deviations. Any zero standard deviation (constant
feature) is replaced by `1f0` to avoid division by zero when normalizing.
Intended to be called on the training split only, to prevent data leakage.
"""
function compute_feature_std(dataset)
    all_features = hcat([sample.x for sample in dataset]...)  # 20 × (total edges)
    stds = vec(std(all_features; dims=2))                      # length-20
    stds[stds .== 0.0f0] .= 1.0f0                                  # avoid division by zero
    return Float32.(stds)
end

"""
$TYPEDSIGNATURES

Draw `(district_μ, district_σ)` for every district from `default_district_μ` and
`default_district_σ`, add them to `instance_sample.context`, and compute the per-edge
feature matrix `x`.
"""
function Utils.generate_context(
    bench::ContextualStochasticVehicleSchedulingBenchmark,
    rng::AbstractRNG,
    instance_sample::DataSample,
)
    instance = instance_sample.context.instance
    city = instance.city
    nb_districts = length(city.districts)
    district_μ = rand(rng, default_district_μ, nb_districts)
    district_σ = rand(rng, default_district_σ, nb_districts)
    # Draw storm hotspot from occupied districts only (districts containing ≥1 task
    # start_point), guaranteeing the storm always affects at least one arc.
    tasks_jobs = city.tasks[2:(end - 1)]
    lin = LinearIndices(city.districts)
    occupied = unique([lin[get_district(t.start_point, city)...] for t in tasks_jobs])
    storm_district = rand(rng, occupied)
    x = compute_contextual_slack_features(
        city,
        instance.graph,
        district_μ,
        district_σ,
        storm_district,
        bench.p_storm,
        bench.storm_multiplier,
        rng;
        nb_feat_scenarios=bench.nb_feat_scenarios,
    )
    return DataSample(;
        x,
        instance_sample.context...,
        district_μ=district_μ,
        district_σ=district_σ,
        storm_district=storm_district,
        p_storm=bench.p_storm,
        storm_multiplier=bench.storm_multiplier,
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
    storm_district::Int,
    p_storm::Float64,
    storm_multiplier::Float64,
    kwargs...,
)
    city = instance.city
    @assert !isnothing(city) "contextual SVS `generate_scenario` requires `store_city=true`"
    lin = LinearIndices(city.districts)
    storm_active = rand(rng) < p_storm
    district_delay_fn =
        (x, y) -> begin
            i = lin[x, y]
            effective_μ = if (storm_active && i == storm_district)
                district_μ[i] + log(storm_multiplier)
            else
                district_μ[i]
            end
            return LogNormal(effective_μ, district_σ[i])
        end
    return draw_scenario(city, instance.graph, rng; district_delay_fn, storm_active)
end

"""
$TYPEDSIGNATURES

Linear model mapping the `20×ne(graph)` per-edge feature matrix to a length-`ne(graph)`
positive score vector. Architecture: `Chain(Dense(20 => 1; bias=false), vec)`.
"""
function Utils.generate_statistical_model(
    ::ContextualStochasticVehicleSchedulingBenchmark; seed=nothing
)
    Random.seed!(seed)
    return Chain(Dense(20 => 1; bias=false), vec)
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
