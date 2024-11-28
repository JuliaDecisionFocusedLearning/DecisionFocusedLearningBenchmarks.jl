"""
$TYPEDSIGNATURES

Compute the achieved travel time of scenario `scenario` from `old_task_index` to `new_task_index`.
"""
function get_perturbed_travel_time(
    city::City, old_task_index::Int, new_task_index::Int, scenario::Int
)
    old_task = city.tasks[old_task_index]
    new_task = city.tasks[new_task_index]

    origin_x, origin_y = get_district(old_task.end_point, city)
    destination_x, destination_y = get_district(new_task.start_point, city)

    ξ₁ = old_task.scenario_end_time[scenario]
    ξ₂ = ξ₁ + city.districts[origin_x, origin_y].scenario_delay[scenario, hour_of(ξ₁)]
    ξ₃ =
        ξ₂ +
        distance(old_task.end_point, new_task.start_point) +
        city.scenario_inter_area_factor[scenario, hour_of(ξ₂)]
    return ξ₃ + city.districts[destination_x, destination_y].scenario_delay[
        scenario, hour_of(ξ₃)
    ] - ξ₁
end

"""
$TYPEDSIGNATURES

Compute slack for features.
"""
function compute_slacks(city::City, old_task_index::Int, new_task_index::Int)
    old_task = city.tasks[old_task_index]
    new_task = city.tasks[new_task_index]

    travel_time = distance(old_task.end_point, new_task.start_point)
    perturbed_end_times = old_task.scenario_end_time
    perturbed_start_times = new_task.scenario_start_time

    return perturbed_start_times .- (perturbed_end_times .+ travel_time)
end

"""
$TYPEDSIGNATURES

Returns a matrix of features of size (20, nb_edges).
For each edge, compute the following features (in the same order):
- travel time
- vehicle_cost if edge is connected to source, else 0
- 9 deciles of the slack
- cumulative probability distribution of the slack evaluated in [-100, -50, -20, -10, 0, 10, 50, 200, 500]
"""
function compute_features(city::City)
    graph = create_VSP_graph(city)

    cumul = [-100, -50, -20, -10, 0, 10, 50, 200, 500]
    nb_features = 2 + 9 + length(cumul)
    features = zeros(nb_features, ne(graph))

    # features indices
    travel_time_index = 1
    connected_to_source_index = 2
    slack_deciles_indices = 3:11
    slack_cumulative_distribution_indices = 12:nb_features

    for (i, edge) in enumerate(edges(graph))
        # compute travel time
        features[travel_time_index, i] = distance(
            city.tasks[src(edge)].end_point, city.tasks[dst(edge)].start_point
        )
        # if edge connected to source node
        features[connected_to_source_index, i] = src(edge) == 1 ? city.vehicle_cost : 0.0

        # slack related features
        slacks = compute_slacks(city, src(edge), dst(edge))
        # compute deciles
        features[slack_deciles_indices, i] = quantile(slacks, [0.1 * i for i in 1:9])
        # compute cumulative distribution
        features[slack_cumulative_distribution_indices, i] = [
            mean(slacks .<= x) for x in cumul
        ]
    end
    return features
end

"""
$TYPEDSIGNATURES

Compute slack for instance.
TODO: differentiate from other method
"""
function compute_slacks(city::City, graph::AbstractGraph)
    (; tasks) = city
    N = nv(graph)
    slack_list = [
        [
            (dst(e) < N ? tasks[dst(e)].scenario_start_time[ω] : Inf) -
            (tasks[src(e)].end_time + get_perturbed_travel_time(city, src(e), dst(e), ω))
            for ω in 1:get_nb_scenarios(city)
        ] for e in edges(graph)
    ]
    I = [src(e) for e in edges(graph)]
    J = [dst(e) for e in edges(graph)]
    return sparse(I, J, slack_list)
end

"""
$TYPEDSIGNATURES

Compute delays for instance.
"""
function compute_delays(city::City)
    nb_tasks = get_nb_tasks(city)
    nb_scenarios = get_nb_scenarios(city)
    ε = zeros(nb_tasks, nb_scenarios)
    for (index, task) in enumerate(city.tasks)
        ε[index, :] .= task.scenario_end_time .- task.end_time
    end
    return ε
end

"""
$TYPEDSIGNATURES

Returns the number of tasks in city.
"""
function get_nb_tasks(city::City)
    return length(city.tasks)
end

"""
$TYPEDSIGNATURES

Returns the number of scenarios in city.
"""
function get_nb_scenarios(city::City)
    return size(city.scenario_inter_area_factor, 1)
end
