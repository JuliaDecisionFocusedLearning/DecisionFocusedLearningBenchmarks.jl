"""
$TYPEDEF

Instance data structure for the dynamic vehicle scheduling problem.
"""
@kwdef struct Instance{I<:StaticInstance,T}
    "static instance to sample arriving requests from"
    static_instance::I
    "max number of new requests per epoch (rejection sampling)"
    max_requests_per_epoch::Int = 10
    "time distance between epoch start and routes start"
    Δ_dispatch::T = 1.0
    "duration of each epoch"
    epoch_duration::T = 1.0
    "last epoch index"
    last_epoch::Int
    # "seed for customer sampling"
    # seed::S
end

function Instance(
    static_instance::StaticInstance;
    max_requests_per_epoch::Int=10,
    Δ_dispatch::Float64=1.0,
    epoch_duration::Float64=1.0,
)
    last_epoch = trunc(
        Int,
        (
            maximum(static_instance.start_time) - minimum(static_instance.duration[1, :]) -
            Δ_dispatch
        ) / epoch_duration,
    )
    return Instance(;
        static_instance=static_instance,
        max_requests_per_epoch=max_requests_per_epoch,
        Δ_dispatch=Δ_dispatch,
        epoch_duration=epoch_duration,
        last_epoch=last_epoch,
    )
end

Δ_dispatch(instance::Instance) = instance.Δ_dispatch
epoch_duration(instance::Instance) = instance.epoch_duration
last_epoch(instance::Instance) = instance.last_epoch
max_requests_per_epoch(instance::Instance) = instance.max_requests_per_epoch
# static_instance(instance::Instance) = instance.static_instance

# duration(instance::Instance) = duration(instance.static_instance)
# service_time(instance::Instance) = service_time(instance.static_instance)
# coordinate(instance::Instance) = coordinate(instance.static_instance)
# start_time(instance::Instance) = start_time(instance.static_instance)
