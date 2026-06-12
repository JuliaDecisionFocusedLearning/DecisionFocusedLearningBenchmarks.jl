"""
$TYPEDEF

Wraps an [`AbstractEnvironment`](@ref) together with a random number generator and an
optional seed, making episodes reproducible.

The wrapper is the single owner of randomness: it threads its own `rng` into the
wrapped environment's [`reset!`](@ref) and [`step!`](@ref), so the wrapped environment
itself never needs to manage a seed. Resetting to the stored seed (see
[`reset_to_initial!`](@ref)) replays the exact same episode, which is what
[`evaluate_policy!`](@ref) relies on for reproducible evaluation.

# Fields
$TYPEDFIELDS
"""
struct SeededEnvironment{E<:AbstractEnvironment,R<:Random.AbstractRNG} <:
       AbstractEnvironment
    "seed (`nothing` if the environment is unseeded)"
    seed::Union{UInt,Nothing}
    "random number generator"
    rng::R
    "wrapped environment"
    env::E
end

function Base.show(io::IO, env::SeededEnvironment)
    show(io, env.env)
    print(io, " (seed=$(env.seed))")
    return nothing
end

"""
$TYPEDSIGNATURES

Wrap `env` in a [`SeededEnvironment`](@ref). By default the generator is a `Xoshiro`
seeded with `seed` (so `seed=nothing` produces an unseeded generator). Pass `rng`
explicitly to supply a different generator.
"""
function SeededEnvironment(env::AbstractEnvironment; seed=nothing, rng=Random.Xoshiro(seed))
    return SeededEnvironment(isnothing(seed) ? nothing : UInt(seed), rng, env)
end

"""
$TYPEDSIGNATURES

Return the stored seed.
"""
function get_seed(env::SeededEnvironment)
    return env.seed
end

"""
$TYPEDSIGNATURES

Return the wrapper's random number generator.
"""
function get_rng(env::SeededEnvironment)
    return env.rng
end

"""
$TYPEDSIGNATURES

Forward to the wrapped environment (see [`is_terminated`](@ref)).
"""
function is_terminated(env::SeededEnvironment)
    return is_terminated(env.env)
end

"""
$TYPEDSIGNATURES

Forward to the wrapped environment (see [`observe`](@ref)).
"""
function observe(env::SeededEnvironment)
    return observe(env.env)
end

"""
$TYPEDSIGNATURES

Reset the wrapped environment using the wrapper's current `rng` state. This advances
the generator (it does not re-seed it), so successive calls produce different episodes.
"""
function reset!(env::SeededEnvironment)
    return reset!(env.env, env.rng)
end

"""
$TYPEDSIGNATURES

Reset the wrapped environment using the provided `rng` instead of the wrapper's own.
"""
function reset!(env::SeededEnvironment, rng::Random.AbstractRNG)
    return reset!(env.env, rng)
end

"""
$TYPEDSIGNATURES

Re-seed the wrapper's `rng` to `seed`, then reset the wrapped environment. Calling this
with a fixed `seed` makes the resulting episode reproducible.
"""
function reset!(env::SeededEnvironment, seed::Integer)
    Random.seed!(env.rng, seed)
    return reset!(env.env, env.rng)
end

"""
$TYPEDSIGNATURES

Equivalent to `reset!(env)`,
i.e. reset from the current `rng` state without re-seeding.
"""
function reset!(env::SeededEnvironment, ::Nothing)
    return reset!(env)
end

"""
$TYPEDSIGNATURES

Reset to the wrapper's initial seed state, replaying the same episode. Equivalent to
`reset!(env, get_seed(env))`.
"""
function reset_to_initial!(env::SeededEnvironment)
    return reset!(env, env.seed)
end

"""
$TYPEDSIGNATURES

Step the wrapped environment, drawing any transition randomness from the wrapper's `rng`.
"""
function step!(env::SeededEnvironment, action)
    return step!(env.env, action, env.rng)
end

"""
$TYPEDSIGNATURES

Step the wrapped environment using the provided `rng` instead of the wrapper's own.
"""
function step!(env::SeededEnvironment, action, rng::Random.AbstractRNG)
    return step!(env.env, action, rng)
end
