"""
$TYPEDEF

Environment data structure for the Dynamic Vehicle Scheduling Problem.

# Fields
$TYPEDFIELDS
"""
@kwdef mutable struct DVSPEnv{C<:DynamicConfig,R<:AbstractRNG,T,S<:VSPState}
    "instance config as a [`DynamicConfig`](@ref)"
    config::C
    "current epoch number"
    current_epoch::Int
    "random number generator"
    rng::R
    "index of each customer in the static instance from the config"
    customer_index::Vector{Int}
    "service time values of each customer"
    service_time::Vector{T}
    "start time values of each customer"
    start_time::Vector{T}
    "1 if the request was already dispatched in a previous epoch, 0 otherwise"
    request_is_dispatched::BitVector
    "epoch index at which each request appearred"
    request_epoch::Vector{Int}
    "current state of environment"
    state::S
end

"""
$TYPEDSIGNATURES

Constructor for [`DVSPEnv`](@ref).
"""
function DVSPEnv(
    static_instance::VSPInstance;
    seed=0,
    max_requests_per_epoch=10,
    Δ_dispatch=1.0,
    epoch_duration=1.0,
)
    first_epoch = 1
    last_epoch = trunc(Int, maximum(static_instance.start_time) / epoch_duration) - 1

    config = DynamicConfig(;
        static_instance,
        max_requests_per_epoch,
        Δ_dispatch,
        epoch_duration,
        seed,
        first_epoch,
        last_epoch,
    )
    return DVSPEnv(;
        config,
        customer_index=[1],
        service_time=[0.0],
        start_time=[0.0],
        request_is_dispatched=falses(1),
        state=VSPState(),
        rng=MersenneTwister(seed),
        current_epoch=0,
        request_epoch=[first_epoch - 1],
    )
end

"""
$TYPEDSIGNATURES

Return the indices of the epochs in the environment.
"""
function get_epoch_indices(env::DVSPEnv)
    return (env.config.first_epoch):(env.config.last_epoch)
end

"""
$TYPEDSIGNATURES

Return the number of epochs in the environment.
"""
function nb_epochs(env::DVSPEnv)
    return length(get_epoch_indices(env))
end

"""
$TYPEDSIGNATURES

Get the current state of the environment.
"""
get_state(env::DVSPEnv) = env.state

"""
$TYPEDSIGNATURES

Get the current time of the environment, i.e. the start time of the current_epoch.
"""
get_time(env::DVSPEnv) = (env.current_epoch - 1) * env.config.epoch_duration

"""
$TYPEDSIGNATURES

Get the planning start time of the environment, i.e. the time at which vehicles routes dispatched in current epoch can depart.
"""
get_planning_start_time(env::DVSPEnv) = get_time(env) + env.config.Δ_dispatch

"""
$TYPEDSIGNATURES

Check if the episode is terminated, i.e. if the current epoch is the last one.
"""
is_terminated(env::DVSPEnv) = env.current_epoch >= env.config.last_epoch

"""
$TYPEDSIGNATURES

Return the total number of locations in the environment history.
"""
nb_locations(env::DVSPEnv) = length(env.customer_index)

"""
$TYPEDSIGNATURES

Return a vector of env location indices that are still undispatched.
"""
get_undispatched_indices(env::DVSPEnv) = (1:nb_locations(env))[.!env.request_is_dispatched]

"""
$TYPEDSIGNATURES

Reset the environment to its initial state.
Also reset the seed if `reset_seed` is set to true.
"""
function reset!(env::DVSPEnv; reset_seed::Bool=true)
    (; config) = env
    env.current_epoch = config.first_epoch - 1
    depot = 1
    env.customer_index = [env.customer_index[depot]]
    env.service_time = [env.service_time[depot]]
    env.start_time = env.start_time[depot:depot]
    env.request_is_dispatched = falses(1)
    env.request_epoch = [env.current_epoch]
    reset_seed && seed!(env.rng, config.seed)
    return nothing
end

"""
$TYPEDSIGNATURES

Internal method that updates the state of the environment to correspond to env info.
This is an internal method and should not be called directly.
"""
function update_state!(env::DVSPEnv)
    (; config) = env
    (; epoch_duration, static_instance, last_epoch) = config
    (; duration) = static_instance
    depot = 1

    planning_start_time = get_planning_start_time(env)

    # Must dispatch
    undispatched_indices = get_undispatched_indices(env)
    # If it's the last epoch, we must dispatch all remaining requests
    is_must_dispatch = undispatched_indices .!= depot
    # Else, only requests unreachable from the depot during next epoch are must dispatch
    if env.current_epoch < last_epoch
        is_must_dispatch =
            planning_start_time .+ epoch_duration .+
            @view(duration[depot, env.customer_index[undispatched_indices]]) .>
            @view(env.start_time[undispatched_indices])
        is_must_dispatch[1] = 0
    end

    is_postponable = falses(length(is_must_dispatch))
    is_postponable[2:end] .= .!is_must_dispatch[2:end]

    epoch_instance = VSPState(;
        instance=VSPInstance(;
            service_time=env.service_time[undispatched_indices],
            start_time=env.start_time[undispatched_indices] .- planning_start_time,  # shift start times to planning start time
            coordinate=static_instance.coordinate[env.customer_index[undispatched_indices]],
            duration=duration[
                env.customer_index[undispatched_indices],
                env.customer_index[undispatched_indices],
            ],
        ),
        is_must_dispatch,
        is_postponable,
    )

    env.state = epoch_instance
    return epoch_instance
end

"""
$TYPEDSIGNATURES

Update `env` by drawing the next epoch and returning a corresponding `EpochInstance`.
"""
function next_epoch!(env::DVSPEnv)
    # Increment epoch number
    env.current_epoch += 1

    # Retrieve useful information
    (; rng, config) = env
    (; max_requests_per_epoch, static_instance) = config
    (; duration, service_time, start_time) = config.static_instance
    depot = 1

    # Draw new requests uniformly from static instance
    N = nb_customers(static_instance)

    planning_start_time = get_planning_start_time(env)

    coordinate_indices = sample_indices(rng, max_requests_per_epoch, N)
    start_time_indices = sample_indices(rng, max_requests_per_epoch, N)
    service_time_indices = sample_indices(rng, max_requests_per_epoch, N)

    # Only keep requests with feasible start times (rejection sampling)
    # i.e. that are reachable from the depot before their start time
    is_feasible =
        planning_start_time .+ duration[depot, coordinate_indices] .<=
        start_time[start_time_indices]

    # Update environment state
    nb_new_requests = sum(is_feasible)

    # Update environment by adding new requests in
    env.customer_index = vcat(env.customer_index, coordinate_indices[is_feasible])
    env.service_time = vcat(
        env.service_time, service_time[service_time_indices[is_feasible]]
    )
    env.start_time = vcat(env.start_time, start_time[start_time_indices[is_feasible]])
    env.request_is_dispatched = vcat(env.request_is_dispatched, falses(nb_new_requests))
    env.request_epoch = vcat(env.request_epoch, fill(env.current_epoch, nb_new_requests))

    # Finally, update the state of the environment with these new requests
    return update_state!(env)
end

"""
$TYPEDSIGNATURES

Transform state routes indices into env route indices.
"""
function env_routes_from_state_routes(env, routes)
    undispatched_indices = get_undispatched_indices(env)
    return [undispatched_indices[route] for route in routes]
end

"""
$TYPEDSIGNATURES

Transform env route indices into state route indices.
"""
function state_route_from_env_routes(env, routes)
    nb_requests = length(env.customer_index)
    undispatched_indices = (1:nb_requests)[.!env.request_is_dispatched]
    global_to_local = zeros(Int, nb_requests)
    for (local_i, global_i) in enumerate(undispatched_indices)
        global_to_local[global_i] = local_i
    end
    return [global_to_local[route] for route in routes]
end

"""
$TYPEDSIGNATURES

Apply given `routes` as an action to `env`.

Routes should be given with global indexation.
Use [`env_routes_from_state_routes`](@ref) if needed to convert the indices beforehand.
"""
function apply_decision!(env::DVSPEnv, routes::Vector{Vector{Int}})
    for route in routes
        env.request_is_dispatched[route] .= true
    end
    duration = @view env.config.static_instance.duration[
        env.customer_index, env.customer_index
    ]
    return cost(routes, duration)
end

"""
$TYPEDSIGNATURES

Draw all epochs until the end of the environment, without any actions.
"""
function draw_all_epochs!(env::DVSPEnv; reset_env=true)
    reset_env && reset!(env)
    while !is_terminated(env)
        next_epoch!(env)
    end
end
