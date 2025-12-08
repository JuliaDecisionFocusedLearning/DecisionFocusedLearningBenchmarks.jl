
struct Scenario
    "indices of the new requests in each epoch"
    indices::Vector{Vector{Int}}
    "service times of the new requests in each epoch"
    service_time::Vector{Vector{Float64}}
    "start times of the new requests in each epoch"
    start_time::Vector{Vector{Float64}}
end

function Base.getindex(scenario::Scenario, idx::Integer)
    return (;
        indices=scenario.indices[idx],
        service_time=scenario.service_time[idx],
        start_time=scenario.start_time[idx],
    )
end

function Utils.generate_scenario(
    instance::Instance; seed=nothing, rng::AbstractRNG=MersenneTwister(seed)
)
    (; Δ_dispatch, static_instance, last_epoch, epoch_duration, max_requests_per_epoch) =
        instance
    (; duration, start_time, service_time) = static_instance
    N = customer_count(static_instance)
    depot = 1

    new_indices = Vector{Int}[]
    new_service_time = Vector{Float64}[]
    new_start_time = Vector{Float64}[]

    min_time, max_time = extrema(start_time)
    for epoch in 1:last_epoch
        time = epoch_duration * (epoch - 1) + Δ_dispatch

        coordinate_indices = sample_indices(rng, max_requests_per_epoch, N)
        # sampled_start_time = sample_times(rng, max_requests_per_epoch, min_time, max_time)
        sampled_start_time = sample_times(
            rng, max_requests_per_epoch, max(min_time, time), max_time
        )
        # start_time_indices = sample_indices(rng, max_requests_per_epoch, N)
        service_time_indices = sample_indices(rng, max_requests_per_epoch, N)

        is_feasible = time .+ duration[depot, coordinate_indices] .<= sampled_start_time # start_time[start_time_indices]

        push!(new_indices, coordinate_indices[is_feasible])
        push!(new_service_time, service_time[service_time_indices[is_feasible]])
        push!(new_start_time, sampled_start_time[is_feasible])
        # push!(new_start_time, start_time[start_time_indices[is_feasible]])
    end
    return Scenario(new_indices, new_service_time, new_start_time)
end

function Utils.generate_scenario(sample::DataSample; kwargs...)
    return Utils.generate_scenario(sample.instance; kwargs...)
end
