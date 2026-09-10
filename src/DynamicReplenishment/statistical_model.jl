"""
$TYPEDEF

Parametrization of the `η` block, i.e. of the marginal-utility curve of each item.

Writing `x[i] = stock[i] + y[i]` for the post-decision stock level, the objective
contribution of item `i` in [`replenishment_problem`](@ref) is

    θ[i] y[i] - Σ_{k ≥ 1} η[i][k] max(0, x[i] - k + 1),

so the marginal utility of its `j`-th unit is

    m_i(j) = θ[i] - Σ_{k=1}^{j} η[i][k].

`m_i(j) - m_i(j+1) = η[i][j+1] ≥ 0` (softplus), so `m_i` is non-increasing: the
  curve is concave, the threshold is where it crosses 0. The constraints `z[i,j] ≥ z[i,j+1]` are redundant at the optimum;

"""
abstract type EtaParametrization end

"""
The most expressive model: the marginal utility of each unit `m_i(j) = θ[i] - Σ_{k≤j} η[i][k]` drops by `η[i][j+1]` at each level: it is concave and piecewise constant.
"""
struct PiecewiseConstantEta <: EtaParametrization end

"Ablation: `η ≡ 0`, so `m_i ≡ θ[i]` and no threshold can be expressed (bang-bang)."
struct NoEta <: EtaParametrization end

"Intercept and a single slope per item: `m_i(j) = θ[i] - a[i] (j - 1)`, 2 parameters/item."
struct SlopeEta <: EtaParametrization end

"Intercept and one slope increment per level, without the first-unit penalty of [`PiecewiseConstantEta`](@ref)."
struct LevelSlopeEta <: EtaParametrization end

"""
$TYPEDEF

# Fields
$TYPEDFIELDS
"""
@kwdef struct StatisticalModel{L1,L2,P<:EtaParametrization}
    "replenishment reward"
    θ_model::L1
    "stock penalization"
    η_model::L2
    "how the `η` block is assembled, see [`EtaParametrization`](@ref)"
    parametrization::P = PiecewiseConstantEta()
end

@layer StatisticalModel trainable = (θ_model, η_model)

"""
$TYPEDSIGNATURES

"""
function Utils.generate_statistical_model(
    b::DynamicReplenishmentBenchmark;
    seed=nothing,
    parametrization=PiecewiseConstantEta(),
    kwargs...,
)
    isnothing(seed) || seed!(seed)
    θ_model = Chain(Dense(item_features_size(b) => 1))
    η_model = Chain(Dense(stock_features_size(b) => 1), softplus)
    return StatisticalModel(; θ_model, η_model, parametrization)
end

function (m::StatisticalModel)(x)
    item_ids = @view x[end, :]
    starts = [findfirst(==(i), item_ids) for i in 1:maximum(Int, item_ids)]

    # the stock block and the trailing item identifier sit on top of the item block
    nb_item_features = size(x, 1) - (NB_STOCK_FEATURES + 1)
    x_features = @view x[1:(end - 1), :]
    x_item = x_features[1:(nb_item_features), starts]
    θ = m.θ_model(x_item)
    return vcat(vec(θ), eta_block(m.parametrization, m.η_model, x_features, starts))
end

"""
$TYPEDSIGNATURES

Item index and level index `j` of each column of the stock-level feature matrix, whose
columns are the candidates `(i, j)` grouped by item, `starts[i]` holding level `j = 1`.
"""
function level_index(n::Int, starts::AbstractVector{Int})
    items = [searchsortedlast(starts, c) for c in 1:n]
    return items, [c - starts[items[c]] + 1 for c in 1:n]
end

"""
$TYPEDSIGNATURES

Assemble the `η` block, of length `sum(ub_per_item)`, from the stock-level features `x`.
"""
eta_block(::PiecewiseConstantEta, η_model, x, starts) = vec(η_model(x))

eta_block(::NoEta, η_model, x, starts) = zeros(eltype(x), size(x, 2))

function eta_block(::SlopeEta, η_model, x, starts)
    # A single slope per item, read on its `j = 1` column, then repeated over its levels:
    # `η[i][k] = a[i]` for every `k ≥ 2` gives `m_i(j) = θ[i] - a[i] (j - 1)`.
    items, levels = level_index(size(x, 2), starts)
    return vec(η_model(x[:, starts]))[items] .* (levels .> 1)
end

function eta_block(::LevelSlopeEta, η_model, x, starts)
    # Same head as `PiecewiseConstantEta`, but level 1 carries no penalty, so `m_i(1) = θ[i]`
    # exactly and
    # every `η[i][k]`, `k ≥ 2`, is a slope increment.
    _, levels = level_index(size(x, 2), starts)
    return vec(η_model(x)) .* (levels .> 1)
end
