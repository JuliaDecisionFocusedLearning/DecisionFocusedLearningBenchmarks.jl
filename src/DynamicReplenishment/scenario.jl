"""
$TYPEDEF

# Fields
$TYPEDFIELDS
"""
@kwdef struct Scenario
    "Perturbed utilities for each customers: utilities[t][k][i] = static_utilities[i] + ε[t][k] where ε[t][k] ~ Gumbel(0, 1): utility of archetype i for customer k at time t"
    utilities::Vector{Vector{Vector{Float64}}}
end

function nb_customers(scenario::Scenario)
    [length(scenario.utilities[t]) for t in 1:length(scenario.utilities)]
end

function Base.getindex(scenario::Scenario, idx::Integer)
    return (; utilities=scenario.utilities[idx])
end

"""
$TYPEDSIGNATURES

Sample a scenario given the customer choice model and static utilities.
"""
function Utils.generate_scenario(
    config::DynamicReplenishmentBenchmark;
    seed=nothing,
    rng::AbstractRNG=Xoshiro(seed),
    temp=1.0,
    random_utility_model=Gumbel(0.0, 1.0),
)
    N = item_count(config)
    T = max_steps(config)
    λ = poisson_arrival_rate(config)
    nb_customers = rand(rng, Poisson(λ), T)
    utilities = [
        [
            config.static_utilities .+ temp * rand(rng, random_utility_model, N+1) for
            _ in 1:nb_customers[t]
        ] for t in 1:T
    ]
    return Scenario(; utilities=utilities)
end
