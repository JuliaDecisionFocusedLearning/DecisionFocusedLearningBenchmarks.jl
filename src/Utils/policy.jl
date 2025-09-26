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
function evaluate_policy!(policy, env::AbstractEnvironment; kwargs...)
    total_reward = 0.0
    local labeled_dataset
    while !is_terminated(env)
        y = policy(env; kwargs...)
        features, state = observe(env)
        if @isdefined labeled_dataset
            push!(
                labeled_dataset,
                DataSample(; x=features, y_true=y, instance=deepcopy(state)),
            )
        else
            labeled_dataset = [DataSample(; x=features, y_true=y, instance=deepcopy(state))]
        end
        reward = step!(env, y)
        total_reward += reward
    end
    return total_reward, labeled_dataset
end

# function evaluate_policy!(policy, envs::Vector{<:AbstractEnvironment}; kwargs...)
#     E = length(envs)
#     rewards = zeros(Float64, E)
#     datasets = map(1:E) do e
#         reward, dataset = evaluate_policy!(policy, envs[e]; kwargs...)
#         rewards[e] = reward
#         return dataset
#     end
#     return rewards, vcat(datasets...)
# end

"""
$TYPEDSIGNATURES

Evaluate the policy on the environment and return the total reward and a dataset of observations.
By default, the environment is reset before running the policy.
"""
function evaluate_policy!(
    policy, env::AbstractEnvironment, episodes::Int; seed=get_seed(env), kwargs...
)
    total_reward = 0.0
    datasets = map(1:episodes) do _i
        reset!(env; reset_rng=(_i == 1))
        reward, dataset = evaluate_policy!(policy, env; kwargs...)
        total_reward += reward
        return dataset
    end
    return total_reward / episodes, vcat(datasets...)
end

"""
$TYPEDSIGNATURES

Run the policy on the environments and return the total rewards and a dataset of observations.
By default, the environments are reset before running the policy.
"""
function evaluate_policy!(
    policy, envs::Vector{<:AbstractEnvironment}, episodes::Int=1; kwargs...
)
    E = length(envs)
    rewards = zeros(Float64, E)
    datasets = map(1:E) do e
        reward, dataset = evaluate_policy!(policy, envs[e], episodes; kwargs...)
        rewards[e] = reward
        return dataset
    end
    return rewards, vcat(datasets...)
end
