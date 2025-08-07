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

Run the policy on the environment and return the total reward and a dataset of observations.
By default, the environment is reset before running the policy.
"""
function run_policy!(policy, env::AbstractEnvironment)
    total_reward = 0.0
    reset!(env; reset_seed=false)
    local labeled_dataset
    while !is_terminated(env)
        y = policy(env)
        features, state = observe(env)
        if @isdefined labeled_dataset
            push!(labeled_dataset, DataSample(; x=features, y_true=y, instance=state))
        else
            labeled_dataset = [DataSample(; x=features, y_true=y, instance=state)]
        end
        reward = step!(env, y)
        total_reward += reward
    end
    return total_reward, labeled_dataset
end

function run_policy!(policy, envs::Vector{<:AbstractEnvironment})
    E = length(envs)
    rewards = zeros(Float64, E)
    datasets = map(1:E) do e
        reward, dataset = run_policy!(policy, envs[e])
        rewards[e] = reward
        return dataset
    end
    return rewards, vcat(datasets...)
end

function run_policy!(policy, env::AbstractEnvironment, episodes::Int; seed=get_seed(env))
    reset!(env; reset_seed=true, seed)
    total_reward = 0.0
    datasets = map(1:episodes) do _i
        reward, dataset = run_policy!(policy, env)
        total_reward += reward
        return dataset
    end
    return total_reward / episodes, vcat(datasets...)
end

function run_policy!(policy, envs::Vector{<:AbstractEnvironment}, episodes::Int)
    E = length(envs)
    rewards = zeros(Float64, E)
    datasets = map(1:E) do e
        reward, dataset = run_policy!(policy, envs[e], episodes)
        rewards[e] = reward
        return dataset
    end
    return rewards, vcat(datasets...)
end
