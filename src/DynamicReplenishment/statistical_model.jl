"""
$TYPEDEF

# Fields
$TYPEDFIELDS
"""
@kwdef struct StatisticalModel{L1,L2}
    "replenishment reward"
    θ_model::L1
    "stock penalization"
    η_model::L2
end

@layer StatisticalModel

"""
$TYPEDSIGNATURES

"""
function Utils.generate_statistical_model(
    b::DynamicReplenishmentBenchmark; seed=nothing, kwargs...
)
    isnothing(seed) || seed!(seed)
    item_features_size = feature_count(b) + 10
    stock_features_size = item_features_size + 8
    θ_model = Chain(Dense(item_features_size => 1))
    η_model = Chain(Dense(stock_features_size => 1), softplus)
    return StatisticalModel(; θ_model, η_model)
end

function (m::StatisticalModel)(x)
    item_ids = @view x[end, :]
    starts = [findfirst(==(i), item_ids) for i in 1:maximum(Int, item_ids)]

    nb_item_features = size(x, 1) - 9
    x_features = @view x[1:(end - 1), :]
    x_item = x_features[1:(nb_item_features), starts]
    θ = m.θ_model(x_item)
    η = m.η_model(x_features)
    return vcat(vec(θ), vec(η))
end
