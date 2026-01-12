"""
$TYPEDEF

Instance of the maintenance problem.

# Fields
$TYPEDFIELDS
"""
@kwdef struct Instance{MaintenanceBenchmark}
    "associated benchmark"
    config::MaintenanceBenchmark
    "starting degradation states"
    starting_state::Vector{Int}
end

"""
$TYPEDSIGNATURES

Generates an instance with random starting degradation states uniformly in [1, n]
"""
function Instance(b::MaintenanceBenchmark, rng::AbstractRNG)
    N = component_count(b)
    n = degradation_levels(b)
    starting_state = rand(rng, 1:n, N)
    return Instance(; config=b, starting_state=starting_state)
end

# Accessor functions
component_count(b::Instance) = component_count(b.config)
maintenance_capacity(b::Instance) = maintenance_capacity(b.config)
degradation_levels(b::Instance) = degradation_levels(b.config)
degradation_probability(b::Instance) = degradation_probability(b.config)
failure_cost(b::Instance) = failure_cost(b.config)
maintenance_cost(b::Instance) = maintenance_cost(b.config)
max_steps(b::Instance) = max_steps(b.config)
starting_state(b::Instance) = b.starting_state
