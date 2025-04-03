"""
$TYPEDSIGNATURES

Retrieve anticipative routes solution from the given MIP solution `y`.
Outputs a set of routes per epoch.
"""
function retrieve_routes_anticipative(y::AbstractArray, dvspenv::DVSPEnv)
    nb_tasks = length(dvspenv.customer_index)
    (; first_epoch, last_epoch) = dvspenv.config
    job_indices = 2:(nb_tasks)
    epoch_indices = first_epoch:last_epoch

    routes = [Vector{Int}[] for t in epoch_indices]
    for t in epoch_indices
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
            push!(routes[t], route)
        end
    end
    return routes
end

"""
$TYPEDSIGNATURES

Solve the anticipative VSP problem for environment `env`.
For this, it uses the current environment history, so make sure that the environment is terminated before calling this method.
"""
function anticipative_solver(env::DVSPEnv; model_builder=highs_model, draw_epochs=true)
    draw_epochs && draw_all_epochs!(env)
    (; customer_index, service_time, start_time, request_epoch) = env
    duration = env.config.static_instance.duration[customer_index, customer_index]
    (; first_epoch, last_epoch, epoch_duration, Δ_dispatch) = env.config

    @assert first_epoch == 1

    model = model_builder()
    set_silent(model)

    nb_nodes = length(customer_index)
    job_indices = 2:nb_nodes
    epoch_indices = first_epoch:last_epoch

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

    # a trip from i can be planned only after request appeared
    for i in job_indices, t in epoch_indices, j in 1:nb_nodes
        if t < request_epoch[i]
            @constraint(model, y[i, j, t] <= 0)
        end
    end

    # a trip from i can be done only before limit date
    for i in job_indices, t in epoch_indices, j in 1:nb_nodes
        if (t - 1) * epoch_duration + duration[1, i] + Δ_dispatch > start_time[i]  # ! this only works if first_epoch = 1
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

    return retrieve_routes_anticipative(value.(y), env)
end
