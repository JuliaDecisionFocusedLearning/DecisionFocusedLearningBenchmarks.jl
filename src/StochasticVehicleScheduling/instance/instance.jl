"""
$TYPEDEF

Instance of the stochastic VSP problem.

# Fields
$TYPEDFIELDS
"""
struct Instance{G<:AbstractGraph,M1<:AbstractMatrix,M2<:AbstractMatrix,F,C}
    "graph computed from `city` with the `create_VSP_graph(city::City)` method"
    graph::G
    "features matrix computed from `city`"
    features::Matrix{F}
    "slack matrix"
    slacks::M1
    "intrinsic delays scenario matrix"
    delays::M2
    "cost of a vehicle"
    vehicle_cost::C
    "cost of one minute delay"
    delay_cost::C
end

"""
$TYPEDSIGNATURES

Return the acyclic directed graph corresponding to `city`.
Each vertex represents a task. Vertices are ordered by start time of corresponding task.
There is an edge from task u to task v the (end time of u + tie distance between u and v <= start time of v).
"""
function create_VSP_graph(city::City)
    # Initialize directed graph
    nb_vertices = city.nb_tasks + 2
    graph = SimpleDiGraph(nb_vertices)
    starting_task = 1
    end_task = nb_vertices
    job_tasks = 2:(city.nb_tasks + 1)

    travel_times = [
        distance(task1.end_point, task2.start_point) for task1 in city.tasks,
        task2 in city.tasks
    ]

    # Create existing edges
    for iorigin in job_tasks
        # link every task to base
        add_edge!(graph, starting_task, iorigin)
        add_edge!(graph, iorigin, end_task)

        for idestination in (iorigin + 1):(city.nb_tasks + 1)
            travel_time = travel_times[iorigin, idestination]
            origin_end_time = city.tasks[iorigin].end_time
            destination_begin_time = city.tasks[idestination].start_time # get_prop(graph, idestination, :task).start_time

            # there is an edge only if we can reach destination from origin before start of task
            if origin_end_time + travel_time <= destination_begin_time
                add_edge!(graph, iorigin, idestination)
            end
        end
    end

    return graph
end

"""
$TYPEDSIGNATURES

Constructor for [`Instance`](@ref).
Build an `Instance` for the stochatsic vehicle scheduling problem, with `nb_tasks` tasks and `nb_scenarios` scenarios.
"""
function Instance(;
    nb_tasks::Int, nb_scenarios::Int, rng::AbstractRNG=Random.default_rng(), kwargs...
)
    city = create_random_city(; rng=rng, nb_tasks, nb_scenarios, kwargs...)
    graph = create_VSP_graph(city)
    features = compute_features(city)
    slacks = compute_slacks(city, graph)
    delays = compute_delays(city)
    return Instance(graph, features, slacks, delays, city.vehicle_cost, city.delay_cost)
end

"""
$TYPEDSIGNATURES

Returns the number of scenarios in instance.
"""
function get_nb_scenarios(instance::Instance)
    return size(instance.delays, 2)
end

"""
$TYPEDSIGNATURES

Returns the number of tasks in `instance`.
"""
get_nb_tasks(instance::Instance) = nv(instance.graph) - 2

"""
$TYPEDSIGNATURES

Returns the feature matrix associated to `instance`.
"""
get_features(instance::Instance) = instance.features
