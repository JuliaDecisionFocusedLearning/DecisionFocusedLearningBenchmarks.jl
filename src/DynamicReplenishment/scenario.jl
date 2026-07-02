"""
$TYPEDEF

# Fields
$TYPEDFIELDS
"""
@kwdef struct Scenario
    "Number of customers per time step"
    nb_customers::Vector{Int}
    "Static utilities"
    static_utilities::Vector{Float64}
    "Perturbed utilities for each customers: utilities[t][k][i] = static_utilities[i] + ε[t][k] where ε[t][k] ~ Gumbel(0, 1): utility of archetype i for customer k at time t"
    utilities::Vector{Vector{Vector{Float64}}}
end

function Base.getindex(scenario::Scenario, idx::Integer)
    return (; nb_customers=scenario.nb_customers[idx], utilities=scenario.utilities[idx])
end

"""
$TYPEDSIGNATURES

Sample a scenario given the customer choice model and static utilities.
"""
function Utils.generate_scenario(
    config::DynamicReplenishmentBenchmark;
    seed=nothing,
    rng::AbstractRNG=MersenneTwister(seed),
    temp=1.0,
    random_utility_model=Gumbel(0.0, 1.0),
)
    Random.seed!(seed)
    N = item_count(config)
    T = max_steps(config)
    λ = poisson_arrival_rate(config)
    nb_customers = rand(rng, Poisson(λ), T)
    full_features = copy(vcat(reshape(prices(config), 1, :), features(config)))
    normalize_features!(full_features; center=true)
    model = customer_choice_model(config)
    static_utilities = model(full_features)
    # add no purchase option
    static_utilities = vcat(static_utilities, 0.0)
    utilities = [
        [
            static_utilities .+ temp * rand(random_utility_model, N+1) for
            _ in 1:nb_customers[t]
        ] for t in 1:T
    ]
    return Scenario(;
        nb_customers=nb_customers, static_utilities=static_utilities, utilities=utilities
    )
end
