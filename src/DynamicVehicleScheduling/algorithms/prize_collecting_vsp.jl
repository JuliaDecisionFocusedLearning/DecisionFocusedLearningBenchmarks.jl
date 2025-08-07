"""
$TYPEDSIGNATURES

Create the acyclic digraph associated with the given VSP `instance`.
"""
function create_graph(instance::StaticInstance)
    (; duration, start_time, service_time) = instance
    # Initialize directed graph
    nb_vertices = location_count(instance)
    graph = SimpleDiGraph(nb_vertices)

    depot = 1  # depot is always index 1
    customers = 2:nb_vertices  # other vertices are customers

    # Create existing edges
    for i₁ in customers
        # link every task to depot
        add_edge!(graph, depot, i₁)
        add_edge!(graph, i₁, depot)

        t₁ = start_time[i₁]
        for i₂ in (i₁ + 1):nb_vertices
            t₂ = start_time[i₂]

            if t₁ <= t₂
                if t₁ + service_time[i₁] + duration[i₁, i₂] <= t₂
                    add_edge!(graph, i₁, i₂)
                end
            else
                if t₂ + service_time[i₂] + duration[i₂, i₁] <= t₁
                    add_edge!(graph, i₂, i₁)
                end
            end
        end
    end

    return graph
end

"""
$TYPEDSIGNATURES

Create the acyclic digraph associated with the given VSP `state`.
"""
function create_graph(state::DVSPState)
    return create_graph(state.state_instance)
end

"""
$TYPEDSIGNATURES

Retrieve routes solution from the given MIP solution `y` matrix and `graph`.
"""
function retrieve_routes(y::AbstractArray, graph::AbstractGraph)
    nb_tasks = nv(graph)
    job_indices = 2:(nb_tasks)
    routes = Vector{Int}[]

    start = [i for i in job_indices if y[1, i] ≈ 1]
    for task in start
        route = Int[]
        current_task = task
        while current_task != 1 # < nb_tasks
            push!(route, current_task)
            local next_task
            for i in outneighbors(graph, current_task)
                if isapprox(y[current_task, i], 1; atol=0.1)
                    next_task = i
                    break
                end
            end
            current_task = next_task
        end
        push!(routes, route)
    end
    return routes
end

"""
$TYPEDSIGNATURES

Solve the Prize Collecting Vehicle Scheduling Problem defined by `instance` and prize vector `θ`.
"""
function prize_collecting_vsp(
    θ::AbstractVector; instance::DVSPState, model_builder=highs_model, kwargs...
)
    (; duration) = instance.state_instance
    graph = create_graph(instance)

    model = model_builder()
    set_silent(model)

    nb_nodes = nv(graph)
    job_indices = 2:(nb_nodes)

    @variable(model, y[i=1:nb_nodes, j=1:nb_nodes; has_edge(graph, i, j)] >= 0)

    θ_ext = fill(0.0, location_count(instance))  # no prize for must dispatch requests, only hard constraints
    θ_ext[instance.is_postponable] .= θ

    @objective(
        model,
        Max,
        sum(
            (θ_ext[dst(edge)] - duration[src(edge), dst(edge)]) * y[src(edge), dst(edge)]
            for edge in edges(graph)
        )
    )
    @constraint(
        model,
        flow[i in 2:nb_nodes],
        sum(y[j, i] for j in inneighbors(graph, i)) ==
            sum(y[i, j] for j in outneighbors(graph, i))
    )
    @constraint(
        model, demand[i in job_indices], sum(y[j, i] for j in inneighbors(graph, i)) <= 1
    )
    # must dispatch constraints
    @constraint(
        model,
        demand_must_dispatch[i in job_indices; instance.is_must_dispatch[i]],
        sum(y[j, i] for j in inneighbors(graph, i)) == 1
    )

    optimize!(model)

    return retrieve_routes(value.(y), graph)
end
