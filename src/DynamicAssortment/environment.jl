"""
$TYPEDEF

Feature 1:d Random static feature
Feature 3: Hype
Feature 4: Satisfaction
Feature 5: Price

# Fields
$TYPEDFIELDS
"""
@kwdef struct Instance{M}
    "customer choice model"
    customer_choice_model::M = Chain(Dense([0.3 0.5 0.6 -0.4 -0.8 0.0]), vec)
    "number of items"
    N::Int = 20
    "dimension of feature vectors (in addition to hype, satisfaction, and price)"
    d::Int = 2
    "assortment size constraint"
    K::Int = 4
    "number of steps per episode"
    max_steps::Int = 80
    "flags if the environment is endogenous"
    endogenous::Bool = true
end

@kwdef mutable struct Environment{R<:AbstractRNG} <: AbstractEnv
    "associated instance"
    instance::Instance
    "current step"
    step::Int
    "purchase history"
    purchase_hist::Vector{Int}
    "rng"
    rng::R
    "seed for RNG"
    seed::Int
    "customer utility for each item"
    utility::Vector{Float64}
    "prices for each item"
    prices::Vector{Float64}
    "current full features"
    features::Matrix{Float64}
    "starting satisfaction + hype features"
    start_features::Matrix{Float64}
    "satisfaction + hype feature change from the last step"
    d_features::Matrix{Float64}
end

function Environment(
    instance::Instance; seed::Int=0, rng::AbstractRNG=MersenneTwister(seed)
)
    return Environment(;
        instance=instance,
        step=1,
        purchase_hist=Int[],
        rng=rng,
        seed=seed,
        utility=zeros(instance.N),
        prices=zeros(instance.N + 1),
        features=zeros(instance.d + 4, instance.N),
        start_features=zeros(2, instance.N),
        d_features=zeros(2, instance.N),
    )
end

## Basic operations of environment

# Reset the environment
function CommonRLInterface.reset!(env::Environment; reset_seed=false, seed=env.seed)
    env.seed = seed
    if reset_seed
        Random.seed!(env.rng, env.seed)
    end
    (; d, N, customer_choice_model) = env.instance
    features = rand(env.rng, Uniform(1.0, 10.0), (d + 3, N))
    env.prices = vcat(features[end, :], 0.0)
    features = vcat(features, ones(1, N))
    env.d_features .= 0.0
    env.step = 1
    env.utility .= customer_choice_model(features)
    env.features .= features
    env.start_features .= features[(d + 1):(d + 2), :]
    env.purchase_hist = Int[]
    return nothing
end

# Update the hype vector
function hype_update!(env::Environment)
    hype_vector = ones(env.instance.N)
    env.purchase_hist[end] != 0 ? hype_vector[env.purchase_hist[end]] += 0.02 : nothing
    if length(env.purchase_hist) >= 2
        if env.purchase_hist[end - 1] != 0
            hype_vector[env.purchase_hist[end - 1]] -= 0.005
        else
            nothing
        end
        if length(env.purchase_hist) >= 3
            if env.purchase_hist[end - 2] != 0
                hype_vector[env.purchase_hist[end - 2]] -= 0.005
            else
                nothing
            end
            if length(env.purchase_hist) >= 4
                if env.purchase_hist[end - 3] != 0
                    hype_vector[env.purchase_hist[end - 3]] -= 0.005
                else
                    nothing
                end
                if length(env.purchase_hist) >= 5
                    if env.purchase_hist[end - 4] != 0
                        hype_vector[env.purchase_hist[end - 4]] -= 0.005
                    else
                        nothing
                    end
                end
            end
        end
    end
    return hype_vector
end

# Step function
function step!(env::Environment, item)
    old_features = copy(env.features)
    push!(env.purchase_hist, item)
    if env.instance.endogenous
        hype_vector = hype_update!(env)
        env.features[3, :] .*= hype_vector
        item != 0 ? env.features[4, item] *= 1.01 : nothing
        env.features[6, :] .+= 9 / env.instance.max_steps # ??
    end
    env.d_features = env.features[3:4, :] - old_features[3:4, :] # ! hardcoded everywhere :(
    env.step += 1
    return nothing
end

# Choice probabilities
function choice_probabilities(env::Environment, S)
    θ = env.utility
    exp_values = [exp(θ[i]) * S[i] for i in 1:(env.instance.N)]
    denominator = 1 + sum(exp_values)
    probs = [exp_values[i] / denominator for i in 1:(env.instance.N)]
    push!(probs, 1 / denominator) # Probability of no purchase
    return probs
end

# Purchase decision
function purchase!(env::Environment, S)
    r = env.prices
    probs = choice_probabilities(env, S)
    item = rand(env.rng, Categorical(probs))
    item == env.instance.N + 1 ? item = 0 : item  # TODO: cleanup this, not really needed and confusing
    item != 0 ? revenue = r[item] : revenue = 0.0
    return item, revenue
end

# enumerate all possible assortments of size K and return the best one
# ? can't we do better than that, probably
function expert_solution(env::Environment)
    r = env.prices
    local best_S
    best_revenue = 0.0
    for S in combinations(1:(env.instance.N), env.instance.K)
        S_vec = zeros(env.instance.N)
        S_vec[S] .= 1.0
        probs = choice_probabilities(env, S_vec)
        expected_revenue = dot(probs, r)
        if expected_revenue > best_revenue
            best_S, best_revenue = S_vec, expected_revenue
        end
    end
    return best_S
end

# DAP CO-layer
function DAP_optimization(θ; instance::Instance)
    solution = partialsortperm(θ, 1:(instance.K); rev=true) # It never makes sense not to show k items
    S = zeros(instance.N)
    S[solution] .= 1
    return S
end

## Solution functions

# Anticipative (fixed)
function expert_policy(env::Environment, episodes; first_seed=1, use_oracle=false)
    dataset = []
    rev_global = Float64[]
    for i in 1:episodes
        rev_episode = 0.0
        CommonRLInterface.reset!(env; seed=first_seed - 1 + i, reset_seed=true)
        done = false
        training_instances = []
        while !done
            S = expert_solution(env)

            delta_features = env.features[3:4, :] .- env.start_features  # ! hardcoded
            feature_vector = vcat(env.features, env.d_features, delta_features)
            push!(training_instances, (features=feature_vector, S_t=S))

            item, revenue = purchase!(env, S)
            rev_episode += revenue
            step!(env, item)

            env.step > env.instance.max_steps ? done = true : done = false
        end
        push!(rev_global, rev_episode)
        push!(dataset, training_instances)
    end
    return mean(rev_global), rev_global, dataset
end

# Greedy heuristic
function model_greedy(features)
    model = Chain(Dense([0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0]), vec)
    return model(features)
end

# Random heuristic
function model_random(features)
    rand_seed = Int(round(sum(features)))
    return rand(MersenneTwister(rand_seed), Uniform(0.0, 1.0), size(features)[2])
end

# Episode generation
function generate_episode(env::Environment, model, customer_model, sigma, random_seed)
    buffer = []
    start_features, d_features = reset!(env; seed=random_seed)
    features = copy(start_features)
    done = false
    while !done
        delta_features = features[3:4, :] .- start_features[3:4, :]
        r = features[5, :]
        feature_vector = vcat(features, d_features, delta_features)
        θ = model(feature_vector)
        η = rand(MersenneTwister(random_seed * env.step), p(θ, sigma), 1)[:, 1]
        S = DAP_optimization(η; instance=env.instance)
        θ_0 = customer_model(features)
        item, revenue = purchase!(env, S)
        features, d_features = step!(env, features, item)
        feat_next = vcat(features, d_features, features[3:4, :] .- start_features[3:4, :])
        push!(
            buffer,
            (
                t=env.step - 1,
                feat_t=feature_vector,
                theta=θ,
                eta=η,
                S_t=S,
                a_T=item,
                rev_t=revenue,
                ret_t=0.0,
                feat_next=feat_next,
            ),
        )
        count(!iszero, inventory) < env.instance.K ? break : nothing
        env.step > env.instance.max_steps ? done = true : done = false
    end
    for i in (length(buffer) - 1):-1:1
        if i == length(buffer) - 1
            ret = buffer[i].rev_t
        else
            ret = buffer[i].rev_t + 0.99 * buffer[i + 1].ret_t
        end
        traj = buffer[i]
        traj_updated = (; traj..., ret_t=ret)
        buffer[i] = traj_updated
    end
    return buffer
end
