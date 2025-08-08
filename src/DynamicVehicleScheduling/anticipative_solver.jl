"""
$TYPEDSIGNATURES

Retrieve anticipative routes solution from the given MIP solution `y`.
Outputs a set of routes per epoch.
"""
function retrieve_routes_anticipative(
    y::AbstractArray, dvspenv::DVSPEnv, customer_index, epoch_indices
)
    nb_tasks = length(customer_index)
    # first_epoch = 1
    # (; last_epoch) = dvspenv.instance
    job_indices = 2:(nb_tasks)
    # epoch_indices = first_epoch:last_epoch

    routes = [Vector{Int}[] for _ in epoch_indices]
    for (i, t) in enumerate(epoch_indices)
        start = [i for i in job_indices if y[1, i, t] ≈ 1]
        for task in start
            route = Int[]
            current_task = task
            while current_task != 1 # < nb_tasks
                push!(route, current_task)
                local next_task
                for i in 1:nb_tasks
                    if isapprox(y[current_task, i, t], 1; atol=0.1)
                        next_task = i
                        break
                    end
                end
                current_task = next_task
            end
            push!(routes[i], route)
        end
    end
    return routes
end

"""
$TYPEDSIGNATURES

Solve the anticipative VSP problem for environment `env`.
For this, it uses the current environment history, so make sure that the environment is terminated before calling this method.
"""
function anticipative_solver(
    env::DVSPEnv,
    scenario=env.scenario;
    model_builder=highs_model,
    two_dimensional_features=env.instance.two_dimensional_features,
    reset_env=true,
    nb_epochs=typemax(Int),
)
    reset_env && reset!(env; reset_seed=true)

    start_epoch = current_epoch(env)
    end_epoch = min(last_epoch(env), start_epoch + nb_epochs - 1)
    T = start_epoch:end_epoch

    request_epoch = [0]
    for t in T
        request_epoch = vcat(request_epoch, fill(t, length(scenario.indices[t])))
    end
    customer_index = vcat(1, scenario.indices[T]...)
    service_time = vcat(0.0, scenario.service_time[T]...)
    start_time = vcat(0.0, scenario.start_time[T]...)

    duration = env.instance.static_instance.duration[customer_index, customer_index]
    (; epoch_duration, Δ_dispatch) = env.instance

    model = model_builder()
    set_silent(model)

    nb_nodes = length(customer_index)
    job_indices = 2:nb_nodes
    epoch_indices = T#first_epoch:last_epoch

    @variable(model, y[i=1:nb_nodes, j=1:nb_nodes, t=epoch_indices]; binary=true)

    @objective(
        model,
        Max,
        sum(
            -duration[i, j] * y[i, j, t] for i in 1:nb_nodes, j in 1:nb_nodes,
            t in epoch_indices
        )
    )

    # flow constraint per epoch
    for t in epoch_indices, i in 1:nb_nodes
        @constraint(
            model,
            sum(y[j, i, t] for j in 1:nb_nodes) == sum(y[i, j, t] for j in 1:nb_nodes)
        )
    end

    # each task must be done once along the horizon
    @constraint(
        model,
        demand[i in job_indices],
        sum(y[j, i, t] for j in 1:nb_nodes, t in epoch_indices) == 1
    )

    # a trip from i can be planned only after request appeared (release times)
    for i in job_indices, t in epoch_indices, j in 1:nb_nodes
        if t < request_epoch[i]
            @constraint(model, y[i, j, t] <= 0)
        end
    end

    # a trip from i can be done only before limit date
    for i in job_indices, t in epoch_indices, j in 1:nb_nodes
        if (t - 1) * epoch_duration + duration[1, i] + Δ_dispatch > start_time[i]
            @constraint(model, y[i, j, t] <= 0)
        end
    end

    # trips can be planned if start, service and transport times enable it
    for i in job_indices, t in epoch_indices, j in job_indices
        if start_time[i] <= start_time[j]
            if start_time[i] + service_time[i] + duration[i, j] > start_time[j]
                @constraint(model, y[i, j, t] <= 0)
            end
        else
            @constraint(model, y[i, j, t] <= 0)
        end
    end

    optimize!(model)

    obj = JuMP.objective_value(model)
    epoch_routes = retrieve_routes_anticipative(
        value.(y), env, customer_index, epoch_indices
    )

    epoch_indices = Vector{Int}[]
    N = 1
    indices = [1]
    index = 1
    for epoch in 1:last_epoch(env)
        M = length(scenario.indices[epoch])
        indices = vcat(indices, (N + 1):(N + M))
        push!(epoch_indices, copy(indices))
        N = N + M
        if epoch in T
            dispatched = vcat(epoch_routes[index]...)
            index += 1
            indices = setdiff(indices, dispatched)
        end
    end

    indices = vcat(1, scenario.indices...)
    start_time = vcat(0.0, scenario.start_time...)
    service_time = vcat(0.0, scenario.service_time...)

    dataset = map(enumerate(T)) do (i, epoch)
        routes = epoch_routes[i]
        epoch_customers = epoch_indices[epoch]

        y_true =
            VSPSolution(
                Vector{Int}[
                    map(idx -> findfirst(==(idx), epoch_customers), route) for
                    route in routes
                ];
                max_index=length(epoch_customers),
            ).edge_matrix

        location_indices = indices[epoch_customers]
        new_coordinates = env.instance.static_instance.coordinate[location_indices]
        new_start_time = start_time[epoch_customers]
        new_service_time = service_time[epoch_customers]
        new_duration = env.instance.static_instance.duration[
            location_indices, location_indices
        ]
        static_instance = StaticInstance(
            new_coordinates, new_service_time, new_start_time, new_duration
        )

        is_must_dispatch = falses(length(location_indices))
        is_postponable = falses(length(location_indices))

        epoch_duration = env.instance.epoch_duration
        Δ_dispatch = env.instance.Δ_dispatch
        planning_start_time = (epoch - 1) * epoch_duration + Δ_dispatch
        if epoch == last_epoch
            # If we are in the last epoch, all requests must be dispatched
            is_must_dispatch[2:end] .= true
        else
            is_must_dispatch[2:end] .=
                planning_start_time .+ epoch_duration .+ @view(new_duration[1, 2:end]) .> new_start_time[2:end]
        end
        is_postponable[2:end] .= .!is_must_dispatch[2:end]

        state = DVSPState(;
            state_instance=static_instance,
            is_must_dispatch,
            is_postponable,
            location_indices,
            current_epoch=epoch,
        )

        x = if two_dimensional_features
            compute_2D_features(state, env.instance)
        else
            compute_features(state, env.instance)
        end

        return DataSample(; instance=state, y_true, x)
    end

    return obj, dataset
end
