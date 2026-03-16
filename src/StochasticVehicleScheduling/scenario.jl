"""
$TYPEDEF

Represents a single scenario for the stochastic vehicle scheduling problem.

# Fields
$TYPEDFIELDS
"""
struct VSPScenario
    "delays per task (length = nb_tasks + 2): scenario_end_time - nominal_end_time"
    delays::Vector{Float64}
    "scalar slack per edge for this scenario"
    slacks::SparseMatrixCSC{Float64,Int}
end

"""
$TYPEDSIGNATURES

Display a compact summary of a [`VSPScenario`](@ref): number of tasks and edges.
"""
function Base.show(io::IO, s::VSPScenario)
    return print(io, "VSPScenario($(length(s.delays) - 2) tasks)")
end

"""
$TYPEDSIGNATURES

Draw a single fresh scenario from the city's random distributions,
independently of the stored scenario draws in the `City` struct.
"""
function draw_scenario(city::City, graph::AbstractGraph, rng::AbstractRNG)
    tasks = city.tasks
    N = length(tasks)

    # 1. Draw inter-area factors for 24 hours (single scenario)
    inter_area = zeros(24)
    previous = 0.0
    for h in 1:24
        previous = (previous + 0.1) * rand(rng, city.random_inter_area_factor)
        inter_area[h] = previous
    end

    # 2. Draw district delays for each district and 24 hours (single scenario)
    nb_per_edge = size(city.districts, 1)
    district_delays = [zeros(24) for _ in 1:nb_per_edge, _ in 1:nb_per_edge]
    for x in 1:nb_per_edge
        for y in 1:nb_per_edge
            prev = 0.0
            for h in 1:24
                prev = scenario_next_delay(prev, city.districts[x, y].random_delay, rng)
                district_delays[x, y][h] = prev
            end
        end
    end

    # 3. Draw task start times (single scenario per task)
    scenario_start_time = [t.start_time + rand(rng, t.random_delay) for t in tasks]

    # 4. Compute task end times for job tasks (indices 2:(N-1))
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

    # 5. Compute delays: scenario_end_time - nominal_end_time
    delays = scenario_end_time .- [t.end_time for t in tasks]

    # 6. Compute scalar slack for each edge in this scenario
    I_idx = [src(e) for e in edges(graph)]
    J_idx = [dst(e) for e in edges(graph)]
    slack_vals = map(edges(graph)) do e
        u = src(e)
        v = dst(e)
        origin_x, origin_y = get_district(tasks[u].end_point, city)
        dest_x, dest_y = get_district(tasks[v].start_point, city)
        ξ₁ = scenario_end_time[u]
        ξ₂ = ξ₁ + district_delays[origin_x, origin_y][hour_of(ξ₁)]
        ξ₃ =
            ξ₂ +
            distance(tasks[u].end_point, tasks[v].start_point) +
            inter_area[hour_of(ξ₂)]
        perturbed_arrival = ξ₃ + district_delays[dest_x, dest_y][hour_of(ξ₃)]
        perturbed_travel_time = perturbed_arrival - ξ₁
        return (v < N ? scenario_start_time[v] : Inf) -
               (tasks[u].end_time + perturbed_travel_time)
    end
    slacks = sparse(I_idx, J_idx, slack_vals, N, N)

    return VSPScenario(delays, slacks)
end

"""
$TYPEDSIGNATURES

Build a stochastic [`Instance`](@ref) from a base instance and a vector of fresh
[`VSPScenario`](@ref)s. Each scenario contributes one column to the `intrinsic_delays`
matrix and one entry per edge to the `slacks` sparse matrix.
"""
function build_stochastic_instance(instance::Instance, scenarios::Vector{VSPScenario})
    K = length(scenarios)
    nb_nodes = length(first(scenarios).delays)
    intrinsic_delays = Matrix{Float64}(undef, nb_nodes, K)
    for (k, s) in enumerate(scenarios)
        intrinsic_delays[:, k] = s.delays
    end

    graph = instance.graph
    N = nv(graph)
    I_idx = [src(e) for e in edges(graph)]
    J_idx = [dst(e) for e in edges(graph)]
    slack_vecs = [[scenarios[k].slacks[src(e), dst(e)] for k in 1:K] for e in edges(graph)]
    new_slacks = sparse(I_idx, J_idx, slack_vecs, N, N)

    return Instance(
        graph,
        instance.features,
        new_slacks,
        intrinsic_delays,
        instance.vehicle_cost,
        instance.delay_cost,
        instance.city,
    )
end
