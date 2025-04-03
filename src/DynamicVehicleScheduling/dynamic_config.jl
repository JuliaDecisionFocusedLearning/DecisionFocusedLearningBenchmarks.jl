"""
$TYPEDEF

Config data structures for dynamic vehicle routing and scheduling problems.

# Fields
$TYPEDFIELDS
"""
@kwdef struct DynamicConfig{I,S,T}
    "static instance to sample arriving requests from"
    static_instance::I
    "max number of new requests per epoch (rejection sampling)"
    max_requests_per_epoch::Int = 100
    "time distance between epoch start and routes start"
    Δ_dispatch::T = 3600
    "duration of each epoch"
    epoch_duration::T = 3600
    "first epoch index (time = epoch_duration x first_epoch)"
    first_epoch::Int
    "last epoch index"
    last_epoch::Int
    "seed for customer sampling"
    seed::S
end
