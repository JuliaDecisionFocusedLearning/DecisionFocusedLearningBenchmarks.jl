"""
$TYPEDEF

Abstract type for environments in decision-focused learning benchmarks.
"""
abstract type AbstractEnvironment end

"""
$TYPEDSIGNATURES

Seed accessor for environments.
By default, environments have no seed.
Override this method to provide a seed for the environment.
"""
function get_seed(::AbstractEnvironment)
    return nothing
end

"""
    is_terminated(env::AbstractEnvironment) --> Bool

Check if the environment has reached a terminal state.
"""
function is_terminated end

"""
    observe(env::AbstractEnvironment) --> Tuple

Get the current observation from the environment.
This function should return a tuple of two elements:
    1. An array of features representing the current state of the environment.
    2. An internal state of the environment, which can be used for further processing (return `nothing` if not needed).
"""
function observe end

"""
    reset!(env::AbstractEnvironment; reset_rng::Bool, seed=get_seed(env)) --> Nothing

Reset the environment to its initial state.
If `reset_rng` is true, the random number generator is reset to the given `seed`.
"""
function reset! end

"""
    step!(env::AbstractEnvironment, action) --> Float64

Perform a step in the environment with the given action.
Returns the reward received after taking the action.
This function may also update the internal state of the environment.
If the environment is terminated, it should raise an error.
"""
function step! end
