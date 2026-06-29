"""
$TYPEDEF

Abstract type for environments in decision-focused learning benchmarks.
"""
abstract type AbstractEnvironment end

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
    reset!(env::AbstractEnvironment, rng::Random.AbstractRNG) --> Nothing

Reset the environment to a starting state using the provided random number generator.
"""
function reset! end

"""
    step!(env::AbstractEnvironment, action, rng::Random.AbstractRNG) --> Float64

Perform a step in the environment with the given action.
Returns the reward received after taking the action.
This function may also update the internal state of the environment.
Implementations that use randomness during a transition must draw from the
provided `rng` rather than any internal state. Deterministic implementations may
ignore the `rng` argument.
If the environment is terminated, it should raise an error.
"""
function step! end
