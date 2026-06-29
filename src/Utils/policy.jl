"""
$TYPEDEF

Policy type for decision-focused learning benchmarks.
"""
struct Policy{P}
    "policy name"
    name::String
    "policy description"
    description::String
    "policy run function"
    policy::P
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

Run the policy from the environment's current state (without resetting), using `rng`
for per-step randomness. Returns the total reward and a dataset of observations.
"""
function rollout!(policy, env::AbstractEnvironment, rng::AbstractRNG; kwargs...)
    total_reward = 0.0
    labeled_dataset = DataSample[]
    step = 0
    while !is_terminated(env)
        step += 1
        y = policy(env; kwargs...)
        features, state = observe(env)
        state_copy = deepcopy(state)
        reward = step!(env, y, rng)
        sample = DataSample(; x=features, y=y, instance=state_copy, extra=(; reward, step))
        if isempty(labeled_dataset)
            labeled_dataset = typeof(sample)[sample]
        else
            push!(labeled_dataset, sample)
        end
        total_reward += reward
    end
    return total_reward, labeled_dataset
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
