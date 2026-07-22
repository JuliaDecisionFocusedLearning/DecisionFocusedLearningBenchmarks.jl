"""
$TYPEDEF

Abstract supertype for all policies acting on benchmark `B`. A policy is anything
callable that produces decisions; the `B` type parameter ties it to the benchmark it was
built for, which lets [`single_rollout!`](@ref) and [`rollout!`](@ref) be specialized per
benchmark by dispatching on the policy's type rather than on a separate benchmark
argument.

See [`AbstractRolloutPolicy`](@ref) and [`AbstractFullHorizonPolicy`](@ref) for the two
concrete flavors.
"""
abstract type AbstractPolicy{B<:AbstractBenchmark} end

"""
$TYPEDEF

Abstract supertype for policies that decide one step at a time. Such a policy is called
once per environment step (as `policy(env; kwargs...) -> y`), via
[`single_rollout!`](@ref), which [`rollout!`](@ref) loops until the environment
terminates.

Specialize [`single_rollout!`](@ref) on `AbstractRolloutPolicy{B}` for a given benchmark
`B` to customize what gets recorded in each step's [`DataSample`](@ref) (e.g. extra
`context` or `extra` fields), without reimplementing the rollout loop.
"""
abstract type AbstractRolloutPolicy{B} <: AbstractPolicy{B} end

"""
$TYPEDEF

Abstract supertype for policies that decide the entire episode at once instead of one
step at a time (e.g. an anticipative/offline solver that plans over the full horizon).

A subtype `P <: AbstractFullHorizonPolicy` must be callable as
`(policy::P)(env::AbstractEnvironment, rng::AbstractRNG; kwargs...) -> (total_reward, dataset)`,
where `dataset` is a `Vector{<:DataSample}`. [`rollout!`](@ref) dispatches straight to
this call instead of stepping through [`single_rollout!`](@ref).
"""
abstract type AbstractFullHorizonPolicy{B} <: AbstractPolicy{B} end

"""
$TYPEDEF

Policy type for decision-focused learning benchmarks, tagged with the benchmark type `B`
it was built for (e.g. `Policy{MyBenchmark}("name", "description", f)`). Wrapping a
callable in `Policy{B}` makes it an [`AbstractRolloutPolicy`](@ref)`{B}`, which enables
per-benchmark specialization of [`single_rollout!`](@ref).
"""
struct Policy{B,P} <: AbstractRolloutPolicy{B}
    "policy name"
    name::String
    "policy description"
    description::String
    "policy run function"
    policy::P
end

"""
$TYPEDSIGNATURES

Construct a `Policy{B}` tagged for benchmark type `B`, e.g. `Policy{MyBenchmark}("Greedy", "...", greedy_policy)`.
"""
function Policy{B}(name::String, description::String, policy::P) where {B,P}
    return Policy{B,P}(name, description, policy)
end

function Base.show(io::IO, p::Policy)
    println(io, "$(p.name): $(p.description)")
    return nothing
end
"""
$TYPEDSIGNATURES

Run the policy and get the next decision on the given environment/instance.
"""
function (p::Policy)(args...; kwargs...)
    return p.policy(args...; kwargs...)
end

"""
$TYPEDSIGNATURES

Perform a single step of a rollout: query `policy` for a decision on the current state of
`env`, record the pre-step features/state, apply the decision with `step!`, and package
the result into a [`DataSample`](@ref) with `extra=(; reward)`.

This is the extension point for customizing what a rollout records: add a method
specialized on `policy::AbstractRolloutPolicy{B}` for a given benchmark `B` to add extra
`context` or `extra` fields, without having to reimplement [`rollout!`](@ref) itself.
"""
function single_rollout!(
    policy::AbstractRolloutPolicy, env::AbstractEnvironment, rng::AbstractRNG; kwargs...
)
    y = policy(env; kwargs...)
    features, state = observe(env)
    state_copy = deepcopy(state)
    reward = step!(env, y, rng)
    return DataSample(; x=features, y=y, instance=state_copy, extra=(; reward))
end

"""
$TYPEDSIGNATURES

Run the policy from the environment's current state (without resetting), using `rng`
for per-step randomness. Returns the total reward and a dataset of observations.

Each step delegates to [`single_rollout!`](@ref); specialize that function instead of
this one to customize what gets recorded per step.
"""
function rollout!(
    policy::AbstractRolloutPolicy, env::AbstractEnvironment, rng::AbstractRNG; kwargs...
)
    total_reward = 0.0
    labeled_dataset = DataSample[]
    while !is_terminated(env)
        sample = single_rollout!(policy, env, rng; kwargs...)
        if isempty(labeled_dataset)
            labeled_dataset = typeof(sample)[sample]
        else
            push!(labeled_dataset, sample)
        end
        total_reward += sample.reward
    end
    return total_reward, labeled_dataset
end

"""
$TYPEDSIGNATURES

Run a full-horizon policy: delegate directly to `policy(env, rng; kwargs...)`, which must
return `(total_reward, dataset)` for the whole episode in one call, instead of looping
step by step through [`single_rollout!`](@ref).
"""
function rollout!(
    policy::AbstractFullHorizonPolicy, env::AbstractEnvironment, rng::AbstractRNG; kwargs...
)
    return policy(env, rng; kwargs...)
end

"""
$TYPEDSIGNATURES

Run the policy on a [`SeededEnvironment`](@ref) and return the total reward and a
dataset of observations. The environment is reset to its initial seed state before
running, which makes the rollout reproducible. Pass `seed` to override the wrapper's
stored seed for this evaluation. The wrapper's `rng` is the single source of
randomness threaded into the underlying environment.
"""
function evaluate_policy!(policy, env::SeededEnvironment; seed=nothing, kwargs...)
    isnothing(seed) ? reset_to_initial!(env) : reset!(env, seed)
    return rollout!(policy, env.env, env.rng; kwargs...)
end

"""
$TYPEDSIGNATURES

Evaluate the policy across multiple episodes. The first episode resets to the
initial seed (or to `seed` if provided), subsequent episodes continue from the
wrapper's evolving rng state.
"""
function evaluate_policy!(
    policy, env::SeededEnvironment, episodes::Int; seed=nothing, kwargs...
)
    rewards = zeros(Float64, episodes)
    datasets = map(1:episodes) do _i
        if _i == 1
            isnothing(seed) ? reset_to_initial!(env) : reset!(env, seed)
        else
            reset!(env)
        end
        reward, dataset = rollout!(policy, env.env, env.rng; kwargs...)
        rewards[_i] = reward
        return dataset
    end
    return rewards, datasets
end

"""
$TYPEDSIGNATURES

Run the policy across a collection of [`SeededEnvironment`](@ref)s.
"""
function evaluate_policy!(
    policy, envs::Vector{<:SeededEnvironment}, episodes::Int=1; kwargs...
)
    E = length(envs)
    avg_rewards = zeros(Float64, E)
    datasets = map(1:E) do e
        rewards, datasets = evaluate_policy!(policy, envs[e], episodes; kwargs...)
        avg_rewards[e] = sum(rewards) / episodes
        dataset = vcat(datasets...)
        return dataset
    end
    return avg_rewards, vcat(datasets...)
end
