"""
$TYPEDEF

Environment for the dynamic assortment problem.

# Fields
$TYPEDFIELDS
"""
@kwdef mutable struct Environment{I<:Instance,R<:AbstractRNG,S<:Union{Nothing,Int}} <:
                      AbstractEnv
    "associated instance"
    instance::I
    "current step"
    step::Int
    "purchase history (used to update hype feature)"
    purchase_hist::Vector{Int}
    "rng"
    rng::R
    "seed for RNG"
    seed::S
    "customer utility for each item"
    utility::Vector{Float64}
    "current full features"
    features::Matrix{Float64}
    "satisfaction + hype feature change from the last step"
    d_features::Matrix{Float64}
end

function Environment(instance::Instance; seed=0, rng::AbstractRNG=MersenneTwister(seed))
    N = item_count(instance)
    (; prices, features, starting_hype_and_saturation) = instance
    full_features = vcat(
        reshape(prices[1:(end - 1)], 1, :), starting_hype_and_saturation, features
    )
    model = customer_choice_model(instance)
    env = Environment(;
        instance,
        step=1,
        purchase_hist=Int[],
        rng=rng,
        seed=seed,
        utility=model(full_features),
        features=full_features,
        d_features=zeros(2, N),
    )
    CommonRLInterface.reset!(env; reset_seed=true)
    return env
end

customer_choice_model(b::Environment) = customer_choice_model(b.instance)
item_count(b::Environment) = item_count(b.instance)
feature_count(b::Environment) = feature_count(b.instance)
assortment_size(b::Environment) = assortment_size(b.instance)
max_steps(b::Environment) = max_steps(b.instance)
prices(b::Environment) = b.instance.prices
# features(b::Environment) = b.instance.features
# starting_hype_and_saturation(b::Environment) = b.instance.starting_hype_and_saturation

## Basic operations of environment

# Reset the environment
function CommonRLInterface.reset!(env::Environment; reset_seed=false, seed=env.seed)
    reset_seed && Random.seed!(env.rng, seed)

    env.step = 1

    (; prices, starting_hype_and_saturation, features) = env.instance
    features = vcat(
        reshape(prices[1:(end - 1)], 1, :), starting_hype_and_saturation, features
    )
    env.features .= features

    env.d_features .= 0.0

    model = customer_choice_model(env)
    env.utility .= model(features)

    empty!(env.purchase_hist)
    return nothing
end

function CommonRLInterface.terminated(env::Environment)
    return env.step > max_steps(env)
end

function CommonRLInterface.observe(env::Environment)
    delta_features = env.features[2:3, :] .- env.instance.starting_hype_and_saturation
    return vcat(
        env.features,
        env.d_features,
        delta_features,
        ones(1, item_count(env)) .* (env.step / max_steps(env) * 10),
    ) #./ 10
end

# Compute the hype vector
function hype_update(env::Environment)
    N = item_count(env)
    hype_vector = ones(N)
    hist = env.purchase_hist

    # Define decay factors for each time step
    factors = [0.02, -0.005, -0.005, -0.005, -0.005]

    # Apply updates for the last 5 purchases
    for (i, factor) in enumerate(factors)
        if length(hist) >= i
            item = hist[end - i + 1]
            if item <= N
                hype_vector[item] += factor
            end
        end
    end

    return hype_vector
end

# Step function
function buy_item!(env::Environment, item::Int)
    push!(env.purchase_hist, item)
    env.step += 1

    if is_endogenous(env.instance.config)
        old_features = copy(env.features[2:3, :])
        # update hype feature
        hype_vector = hype_update(env)
        env.features[2, :] .*= hype_vector

        # update saturation feature
        if item <= item_count(env)
            env.features[3, item] *= 1.01
        end

        env.utility .= customer_choice_model(env)(env.features)
        env.d_features = env.features[2:3, :] - old_features
    end
    return nothing
end

# Choice probabilities
function choice_probabilities(env::Environment, S)
    N = item_count(env)
    θ = env.utility
    exp_values = [exp(θ[i]) * S[i] for i in 1:N]
    push!(exp_values, 1.0) # No purchase action
    denominator = sum(exp_values)
    probs = exp_values ./ denominator
    return probs
end

# Purchase decision
function CommonRLInterface.act!(env::Environment, S)
    r = prices(env)
    probs = choice_probabilities(env, S)
    item = rand(env.rng, Categorical(probs))
    reward = r[item]
    buy_item!(env, item)
    return reward
end

## Solution functions
# enumerate all possible assortments of size K and return the best one
function compute_expected_revenue(env::Environment, S)
    r = prices(env)
    probs = choice_probabilities(env, S)
    expected_revenue = dot(probs, r)
    return expected_revenue
end

function expert_solution(env::Environment)
    N = item_count(env)
    K = assortment_size(env)
    best_S = falses(N)
    best_revenue = -1.0
    S_vec = falses(N)
    for S in combinations(1:N, K)
        S_vec .= false
        S_vec[S] .= true
        expected_revenue = compute_expected_revenue(env, S_vec)
        if expected_revenue > best_revenue
            best_S, best_revenue = copy(S_vec), expected_revenue
        end
    end
    return best_S
end

function greedy_solution(env::Environment)
    maximizer = generate_maximizer(env.instance.config)
    return maximizer(prices(env))
end

function run_policy(env::Environment, episodes::Int; first_seed=1, policy=expert_solution)
    dataset = []
    rev_global = Float64[]
    for i in 1:episodes
        rev_episode = 0.0
        CommonRLInterface.reset!(env; seed=first_seed - 1 + i, reset_seed=true)
        training_instances = []
        while !CommonRLInterface.terminated(env)
            S = policy(env)
            features = CommonRLInterface.observe(env)
            push!(training_instances, DataSample(; x=features, y_true=S))
            reward = CommonRLInterface.act!(env, S)
            rev_episode += reward
        end
        push!(rev_global, rev_episode)
        push!(dataset, training_instances)
    end
    return mean(rev_global), rev_global, dataset
end
