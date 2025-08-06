"""
$TYPEDEF

Instance of the dynamic assortment problem.

# Fields
$TYPEDFIELDS
"""
@kwdef struct Instance{B<:DynamicAssortmentBenchmark}
    "associated benchmark"
    config::B
    "item prices (including no purchase action)"
    prices::Vector{Float64}
    "static features, size (d, N)"
    features::Matrix{Float64}
    "starting hype and saturation features, size (2, N)"
    starting_hype_and_saturation::Matrix{Float64}
end

function Instance(b::DynamicAssortmentBenchmark, rng::AbstractRNG)
    N = item_count(b)
    d = feature_count(b)
    prices = vcat(rand(rng, Uniform(1.0, 10.0), N), 0.0) # last price is for no purchase action
    features = rand(rng, Uniform(1.0, 10.0), (d, N))
    starting_hype_and_saturation = rand(rng, Uniform(1.0, 10.0), (2, N))
    return Instance(; config=b, prices, features, starting_hype_and_saturation)
end

customer_choice_model(b::Instance) = customer_choice_model(b.config)
item_count(b::Instance) = item_count(b.config)
feature_count(b::Instance) = feature_count(b.config)
assortment_size(b::Instance) = assortment_size(b.config)
max_steps(b::Instance) = max_steps(b.config)
