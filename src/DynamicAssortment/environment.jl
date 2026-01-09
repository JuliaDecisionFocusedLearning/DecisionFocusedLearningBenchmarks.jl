"""
$TYPEDEF

Environment for the dynamic assortment problem.

# Fields
$TYPEDFIELDS
"""
@kwdef mutable struct Environment{I<:Instance,R<:AbstractRNG,S<:Union{Nothing,Int}} <:
                      Utils.AbstractEnvironment
    "associated instance"
    instance::I
    "current step"
    step::Int
    "purchase history (used to update hype feature)"
    purchase_history::Vector{Int}
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

"""
$TYPEDSIGNATURES

Creates an [`Environment`](@ref) from an [`Instance`](@ref) of the dynamic assortment benchmark.
"""
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
        purchase_history=Int[],
        rng=rng,
        seed=seed,
        utility=model(full_features),
        features=full_features,
        d_features=zeros(2, N),
    )
    Utils.reset!(env; reset_rng=true)
    return env
end

customer_choice_model(env::Environment) = customer_choice_model(env.instance)
item_count(env::Environment) = item_count(env.instance)
feature_count(env::Environment) = feature_count(env.instance)
assortment_size(env::Environment) = assortment_size(env.instance)
max_steps(env::Environment) = max_steps(env.instance)
prices(env::Environment) = prices(env.instance)

"""
$TYPEDSIGNATURES

Compute an hype multiplier vector based on the purchase history.
The hype multiplier (equal to 1 by default) for each item is updated as follows:
- If the item was purchased in the last step, its hype multiplier increases by 0.02.
- If the item was purchased in the last 2 to 5 steps, its hype multiplier decreases by 0.005.
"""
function hype_update(env::Environment)
    N = item_count(env)
    hype_vector = ones(N)
    hist = env.purchase_history

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

"""
$TYPEDSIGNATURES

Updates the environment state after a purchase of `item`.
"""
function buy_item!(env::Environment, item::Int)
    push!(env.purchase_history, item)
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

"""
$TYPEDSIGNATURES

Compute the choice probabilities for each item in `assortment`.
"""
function choice_probabilities(env::Environment, assortment::BitVector)
    N = item_count(env)
    θ = env.utility
    exp_values = [exp(θ[i]) * assortment[i] for i in 1:N]
    push!(exp_values, 1.0) # No purchase action
    denominator = sum(exp_values)
    probs = exp_values ./ denominator
    return probs
end

"""
$TYPEDSIGNATURES

Compute the expected revenue of offering `assortment`.
"""
function compute_expected_revenue(env::Environment, assortment::BitVector)
    r = prices(env)
    probs = choice_probabilities(env, assortment)
    expected_revenue = dot(probs, r)
    return expected_revenue
end

"""
$TYPEDSIGNATURES

Outputs the seed of the environment.
"""
Utils.get_seed(env::Environment) = env.seed

"""
$TYPEDSIGNATURES

Resets the environment to the initial state:
- reset the rng if `reset_rng` is true
- reset the step to 1
- reset the features to the initial features
- reset the change in features to zero
- reset the utility to the initial utility
- clear the purchase history
"""
function Utils.reset!(env::Environment; reset_rng=false, seed=env.seed)
    reset_rng && Random.seed!(env.rng, seed)

    env.step = 1

    (; prices, starting_hype_and_saturation, features) = env.instance
    features = vcat(
        reshape(prices[1:(end - 1)], 1, :), starting_hype_and_saturation, features
    )
    env.features .= features

    env.d_features .= 0.0

    model = customer_choice_model(env)
    env.utility .= model(features)

    empty!(env.purchase_history)
    return nothing
end

"""
$TYPEDSIGNATURES

Checks if the environment has reached the maximum number of steps.
"""
function Utils.is_terminated(env::Environment)
    return env.step > max_steps(env)
end

"""
$TYPEDSIGNATURES

Features observed by the agent at current step, as a concatenation of:
- current full features (including prices, hype, saturation, and static features)
- change in hype and saturation features from the last step
- change in hype and saturation features from the starting state
- normalized current step (divided by max steps and multiplied by 10)
All features are normalized by dividing by 10.

State
Return as a tuple:
- `env.features`: the current feature matrix (feature vector for all items).
- `env.purchase_history`: the purchase history over the most recent steps.
"""
function Utils.observe(env::Environment)
    delta_features = env.features[2:3, :] .- env.instance.starting_hype_and_saturation
    features =
        vcat(
            env.features,
            env.d_features,
            delta_features,
            ones(1, item_count(env)) .* (env.step / max_steps(env) * 10),
        ) ./ 10

    state = (copy(env.features), copy(env.purchase_history))

    return features, state
end

"""
$TYPEDSIGNATURES

Performs one step in the environment given an assortment.
Draw an item according to the customer choice model and updates the environment state.
"""
function Utils.step!(env::Environment, assortment::BitVector)
    @assert !Utils.is_terminated(env) "Environment is terminated, cannot act!"
    r = prices(env)
    probs = choice_probabilities(env, assortment)
    item = rand(env.rng, Categorical(probs))
    reward = r[item]
    buy_item!(env, item)
    return reward
end
