"""
$TYPEDEF

# Fields
$TYPEDFIELDS
"""
@kwdef struct statistical_model{L1,L2}
    "replenishment reward"
    θ_model::L1
    "stock penalization"
    η_model::L2
end

@layer statistical_model

"""
$TYPEDSIGNATURES

"""
function Utils.generate_statistical_model(b::DynamicReplenishmentBenchmark)
    item_features_size = feature_count(b) + 10
    stock_features_size = item_features_size + 8
    θ_model = Chain(Dense(item_features_size => 1))
    η_model = Chain(Dense(stock_features_size => 1), softplus)
    return statistical_model(; θ_model, η_model)
end

"""
$TYPEDSIGNATURES

"""
function (m::statistical_model)(x, N, ub)
    nb_item_features = size(x, 1) - 8              # features are along dim 1
    x_item = x[1:nb_item_features, 1:ub:(N * ub)]  # feature rows, one col per item
    θ = m.θ_model(x_item)
    η = m.η_model(x)
    return vcat(vec(θ), vec(η))
end