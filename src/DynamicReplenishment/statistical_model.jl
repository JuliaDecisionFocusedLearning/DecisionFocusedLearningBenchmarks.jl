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

# Les trois variantes, et la seule chose qui les distingue

Elles ne diffèrent que par le NOMBRE de `η` réellement produits par le modèle :

| | `m_i(j)` | `η` par item |
|---|---|---|
| [`PiecewiseConstantEta`](@ref) (`full`)  | `θ_i − Σ_{k≤j} η_{i,k}` | `ub_i` |
| [`SlopeEta`](@ref) (`slope`)             | `θ_i − j · η_i`         | `1` |
| [`NoEta`](@ref) (`none`)                 | `θ_i`                   | `0` |

`slope` est exactement `full` avec `η_{i,k} ≡ η_i` : la première unité EST
pénalisée (`m_i(1) = θ_i − η_i`), contrairement aux anciennes `SlopeEta` /
`LevelSlopeEta` qui exemptaient le niveau 1 et ont été retirées.

# ⚠️ Pourquoi la longueur de `Θ` change, et pourquoi c'est le point

`Θ = [θ (N) ; η (nb_eta)]`, et `nb_eta` vaut `Σ ub_i`, `N` ou `0` selon la
variante. Une version antérieure gardait `Θ` à pleine longueur et remplissait le
bloc `η` de zéros pour `none` — mais la perte de Fenchel-Young perturbe
`Θ` coordonnée par coordonnée (`PerturbedAdditive` : `θ .+ ε·Z` sur TOUT le
vecteur). Les zéros recevaient donc du bruit, devenaient négatifs une fois sur
deux, et un `η` négatif RÉCOMPENSE le stock : l'ablation « pas de seuil »
s'entraînait en fait contre des cibles portant un seuil aléatoire.

Mesuré avant correction, sur les instances de l'étude, part des tirages où le
bruit sur `η` changeait la décision perturbée :

    ε = 1    n = 10 : 68 %   n = 50 : 94 %
    ε = 10   n = 10 : 96 %   n = 50 : 99 %

En ne produisant que les `η` qui existent, le bruit ne peut plus porter que sur
des degrés de liberté réels. Effet de bord bienvenu : le modèle est plus petit
et le vecteur perturbé plus court.

⚠️ Conséquence : `Θ` n'a plus la même longueur d'une variante à l'autre, donc
[`replenishment_problem`](@ref), [`g`](@ref) et `g_model` doivent connaître la
paramétrisation. Les trois la reçoivent, et [`nb_eta`](@ref) /
[`expand_eta`](@ref) ci-dessous sont leur unique source de vérité : toute
divergence entre eux casserait l'identité `⟨g(y), Θ⟩ = objectif(y)` dont dépend
le gradient de la perte.
"""
abstract type EtaParametrization end

"""
`full` — la variante la plus expressive : un `η` par item et par niveau, donc
`m_i(j) = θ_i − Σ_{k≤j} η_{i,k}`, concave et constante par morceaux.
"""
struct PiecewiseConstantEta <: EtaParametrization end

"""
`none` — ablation : aucun `η` n'est produit, `m_i ≡ θ_i`. L'objectif devient
linéaire en `y`, donc l'optimum est un sommet du polytope des quotas
(bang-bang) et aucun seuil n'est exprimable.
"""
struct NoEta <: EtaParametrization end

"""
`slope` — un seul `η_i` par item, appliqué à TOUS les niveaux :
`m_i(j) = θ_i − j·η_i`. Deux paramètres effectifs par item, `θ_i` et `η_i`.

La pente unique est lue sur la colonne de niveau [`SLOPE_LEVEL`](@ref) du bloc
de l'item.
"""
struct SlopeEta <: EtaParametrization end

"""
Colonne de niveau sur laquelle [`SlopeEta`](@ref) lit la pente unique de chaque
item : `j = 1`, la première du bloc.

⚠️ Le choix naturel serait `j = s_i + 1`, le niveau de la prochaine unité, donc
le point de fonctionnement courant. Il n'est PAS accessible ici : le modèle
statistique est appelé `m(x)` et ne reçoit pas l'état, et `s_i` n'est pas
reconstructible depuis `x` — `ScaledModel` divise les features par un `scale`
que le modèle ne connaît pas, donc `max_quotas_dev = max_quotas[t,i] − j`
n'est plus lisible en entier. Le rendre accessible demanderait soit de faire
passer l'état jusqu'au modèle (donc de toucher la boucle de perte dans
DecisionFocusedLearningAlgorithms), soit d'ajouter une feature de niveau non
normalisée (donc d'invalider tous les modèles archivés).

En pratique l'écart est faible : les features d'item sont identiques sur toutes
les colonnes d'un item, et seules les 12 features de niveau varient. Sur les
instances parcimonieuses (`λ/N = 0.2`) la plupart des items ont `s_i = 0`, pour
lesquels `j = 1` EST `s_i + 1`.
"""
const SLOPE_LEVEL = 1

"""
$TYPEDSIGNATURES

Longueur du bloc `η` de `Θ` — la seule fonction qui la définit.
"""
nb_eta(::PiecewiseConstantEta, ub::AbstractVector{<:Integer}) = sum(ub)
nb_eta(::SlopeEta, ub::AbstractVector{<:Integer}) = length(ub)
nb_eta(::NoEta, ub::AbstractVector{<:Integer}) = 0

"""
$TYPEDSIGNATURES

Décompacte le bloc `η` en ses incréments PAR NIVEAU `η[i][k]`, `k = 1..ub[i]`.

C'est la forme que lisent la fonction objectif et les figures ; `slope` répète
son unique pente sur tous les niveaux, `none` rend des zéros. Ne sert PAS à
construire `Θ` — le modèle, lui, produit la forme compacte.
"""
function expand_eta(::PiecewiseConstantEta, ηc::AbstractVector, ub::AbstractVector{<:Integer})
    out = Vector{Vector{eltype(ηc)}}(undef, length(ub))
    off = 0
    for i in eachindex(ub)
        out[i] = collect(ηc[(off + 1):(off + ub[i])])
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

Assemble le bloc `η` COMPACT depuis les features de niveau `x`.

Sa longueur est [`nb_eta`](@ref), pas `sum(ub)` : c'est tout l'objet de la
refonte, voir [`EtaParametrization`](@ref). `none` ne fait tourner aucune tête
`η`, `slope` ne la fait tourner que sur une colonne par item.
"""
eta_block(::PiecewiseConstantEta, η_model, x, starts) = vec(η_model(x))

# Aucun `η` : on rend un vecteur VIDE, et non des zéros. Un zéro serait une
# coordonnée de `Θ` que la perturbation de Fenchel-Young irait bruiter.
eta_block(::NoEta, η_model, x, starts) = similar(x, 0)

# Une seule colonne par item — donc `N` passes avant au lieu de `sum(ub)`.
eta_block(::SlopeEta, η_model, x, starts) =
    vec(η_model(x[:, starts .+ (SLOPE_LEVEL - 1)]))
