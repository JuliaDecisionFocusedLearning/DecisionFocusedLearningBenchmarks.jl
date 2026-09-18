"""
$TYPEDEF

Parametrization of the `η` block, i.e. of the marginal-utility curve of each item.

Writing `x[i] = stock[i] + y[i]` for the post-decision stock level, the objective
contribution of item `i` in [`replenishment_problem`](@ref) is

    θ[i] y[i] - Σ_j z[i,j] Σ_{k≤j} η[i][k],

so the marginal utility of its `j`-th unit is

    m_i(j) = θ[i] - Σ_{k=1}^{j} η[i][k].

`η ≥ 0` (softplus), so `m_i(j) - m_i(j+1) = η[i][j+1] ≥ 0`: the curve is concave
and the threshold is where it crosses zero.

# The three variants, and the only thing that sets them apart

They differ only in the NUMBER of `η` the model actually produces:

| | `m_i(j)` | `η` per item |
|---|---|---|
| [`PiecewiseConstantEta`](@ref) (`full`)  | `θ_i − Σ_{k≤j} η_{i,k}` | `ub_i` |
| [`SlopeEta`](@ref) (`slope`)             | `θ_i − j · η_i`         | `1` |
| [`NoEta`](@ref) (`none`)                 | `θ_i`                   | `0` |

`slope` is exactly `full` with `η_{i,k} ≡ η_i`: the first unit IS penalized
(`m_i(1) = θ_i − η_i`), unlike the former `SlopeEta` / `LevelSlopeEta` which
exempted level 1 and have since been removed.

# ⚠️ Why the length of `Θ` varies, and why that is the whole point

`Θ = [θ (N) ; η (nb_eta)]`, and `nb_eta` is `Σ ub_i`, `N` or `0` depending on
the variant. An earlier version kept `Θ` at full length and padded the `η` block
with zeros for `none` — but the Fenchel-Young loss perturbs `Θ` coordinate by
coordinate (`PerturbedAdditive`: `θ .+ ε·Z` over the WHOLE vector). The zeros
therefore received noise, went negative half of the time, and a negative `η`
REWARDS stock: the "no threshold" ablation was in fact training against targets
that carried a random threshold.

Measured before the fix, on the instances of the study, share of the draws in
which the noise on `η` changed the perturbed decision:

    ε = 1    n = 10 : 68 %   n = 50 : 94 %
    ε = 10   n = 10 : 96 %   n = 50 : 99 %

By producing only the `η` that exist, the noise can no longer fall anywhere but
on real degrees of freedom. A welcome side effect: the model is smaller and the
perturbed vector shorter.

⚠️ Consequence: `Θ` no longer has the same length across variants, so
[`replenishment_problem`](@ref), [`g`](@ref) and `g_model` must know the
parametrization. All three receive it, and [`nb_eta`](@ref) /
[`expand_eta`](@ref) below are their single source of truth: any divergence
between them would break the identity `⟨g(y), Θ⟩ = objective(y)` that the
gradient of the loss relies on.
"""
abstract type EtaParametrization end

"""
`full` — the most expressive variant: one `η` per item and per level, hence
`m_i(j) = θ_i − Σ_{k≤j} η_{i,k}`, concave and piecewise constant.
"""
struct PiecewiseConstantEta <: EtaParametrization end

"""
`none` — ablation: no `η` is produced at all, `m_i ≡ θ_i`. The objective becomes
linear in `y`, so the optimum is a vertex of the quota polytope (bang-bang) and
no threshold can be expressed.
"""
struct NoEta <: EtaParametrization end

"""
`slope` — a single `η_i` per item, applied to ALL levels:
`m_i(j) = θ_i − j·η_i`. Two effective parameters per item, `θ_i` and `η_i`.

The single slope is read off the level column [`SLOPE_LEVEL`](@ref) of the
item's block.
"""
struct SlopeEta <: EtaParametrization end

"""
Level column on which [`SlopeEta`](@ref) reads the single slope of each item:
`j = 1`, the first one of the block.

⚠️ The natural choice would be `j = s_i + 1`, the level of the next unit, hence
the current operating point. It is NOT reachable here: the statistical model is
called as `m(x)` and never receives the state, and `s_i` cannot be reconstructed
from `x` — `ScaledModel` divides the features by a `scale` the model knows
nothing about, so `max_quotas_dev = max_quotas[t,i] − j` can no longer be read
back as an integer. Making it reachable would mean either threading the state
down to the model (hence touching the loss loop in
DecisionFocusedLearningAlgorithms), or adding an unnormalized level feature
(hence invalidating every archived model).

In practice the gap is small: the item features are identical across all columns
of an item, and only the 12 level features vary. On sparse instances
(`λ/N = 0.2`) most items have `s_i = 0`, for which `j = 1` IS `s_i + 1`.
"""
const SLOPE_LEVEL = 1

"""
$TYPEDSIGNATURES

Length of the `η` block of `Θ` — the only function that defines it.
"""
nb_eta(::PiecewiseConstantEta, ub::AbstractVector{<:Integer}) = sum(ub)
nb_eta(::SlopeEta, ub::AbstractVector{<:Integer}) = length(ub)
nb_eta(::NoEta, ub::AbstractVector{<:Integer}) = 0

"""
$TYPEDSIGNATURES

Unpacks the `η` block into its PER-LEVEL increments `η[i][k]`, `k = 1..ub[i]`.

This is the form read by the objective function and by the figures; `slope`
repeats its single slope across every level, `none` returns zeros. It is NOT
used to build `Θ` — the model itself produces the compact form.
"""
function expand_eta(::PiecewiseConstantEta, ηc::AbstractVector, ub::AbstractVector{<:Integer})
    out = Vector{Vector{eltype(ηc)}}(undef, length(ub))
    off = 0
    for i in eachindex(ub)
        out[i] = collect(ηc[(off+1):(off+ub[i])])
        off += ub[i]
    end
    return out
end
expand_eta(::SlopeEta, ηc::AbstractVector, ub::AbstractVector{<:Integer}) =
    [fill(ηc[i], ub[i]) for i in eachindex(ub)]
expand_eta(::NoEta, ηc::AbstractVector, ub::AbstractVector{<:Integer}) =
    [zeros(eltype(ηc), ub[i]) for i in eachindex(ub)]

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
    x_features = @view x[1:(end-1), :]
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

Assembles the COMPACT `η` block from the level features `x`.

Its length is [`nb_eta`](@ref), not `sum(ub)`: that is the very purpose of the
rework, see [`EtaParametrization`](@ref). `none` runs no `η` head at all,
`slope` runs it on a single column per item.
"""
eta_block(::PiecewiseConstantEta, η_model, x, starts) = vec(η_model(x))

# No `η` at all: return an EMPTY vector rather than zeros. A zero would be a
# coordinate of `Θ` for the Fenchel-Young perturbation to add noise to.
eta_block(::NoEta, η_model, x, starts) = similar(x, 0)

# A single column per item — hence `N` forward passes instead of `sum(ub)`.
eta_block(::SlopeEta, η_model, x, starts) =
    vec(η_model(x[:, starts .+ (SLOPE_LEVEL-1)]))