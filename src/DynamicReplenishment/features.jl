function mean_feature_matrix(state::DRPState, feature_matrix)
    N = item_count(state.config)
    @assert size(feature_matrix, 2) == N
    isempty(feature_matrix) && return zeros(Float64, N)
    return vec(mean(feature_matrix; dims=1))
end

mean_sales_history(state::DRPState) = mean_feature_matrix(state, sales_history(state))
function mean_replenishment_history(state::DRPState)
    return mean_feature_matrix(state, replenishment_history(state))
end
mean_stock_history(state::DRPState) = mean_feature_matrix(state, stock_history(state))

function items_with_positive_stock(state::DRPState)
    isempty(stock_history(state)) && return Int[]
    return [
        i for i in 1:item_count(state.config) if sum(view(stock_history(state), :, i)) > 0
    ]
end

"""
Return mean features of item if it was ever in stock,
return mean features across all items that have had positive stock otherwise.
"""
function item_mean_feature(
    feature_row_means::Vector{Float64}, item::Int, positive_stock_items::Vector{Int}
)
    isempty(positive_stock_items) && return 0.0
    if item ∈ positive_stock_items
        return feature_row_means[item]
    else
        return mean_or_zero(feature_row_means[positive_stock_items])
    end
end

"""
Compute the Days On Lot of the items.
It corresponds to the number of time step an item stays physically in stock before being sold.
It can be negative if the item is sold before arriving.
"""
function compute_dol_item(state::DRPState, item::Int)
    replenishments = replenishment_history(state)  # (t, N)
    sales = sales_history(state)                   # (t, N)
    s0 = stock_ini(state)[item]
    t_now = current_epoch(state)

    repl_item = isempty(replenishments) ? Int[] : replenishments[:, item]
    sales_item = isempty(sales) ? Int[] : sales[:, item]

    cum_arrivals = s0 .+ cumsum(repl_item)   # cum_arrivals[t] = total arrived by end of epoch t
    total_nb_item = s0 + sum(repl_item)

    total_nb_item == 0 && return Float64[]

    cum_sales = cumsum(sales_item)           # cum_sales[t] = total sold by end of epoch t
    total_nb_sales = isempty(cum_sales) ? 0 : cum_sales[end]

    if total_nb_sales == 0
        return fill(Float64(t_now), total_nb_item)
    end

    dols = zeros(Float64, total_nb_item)
    for j in 1:total_nb_item
        date_repl_j = if j <= s0
            1
        else
            t = findfirst(>=(j), cum_arrivals)
            t === nothing ? t_now : t + 1  # arrives the epoch *after* the replenishment lands
        end

        end_date_j = if j <= total_nb_sales
            something(findfirst(>=(j), cum_sales), t_now)
        else
            t_now
        end

        dols[j] = end_date_j - date_repl_j + 1
    end
    return dols
end

# =============================================================================
# Feature sets
#
# `:v1` (default) is the historical set, unchanged.
#
# `:v2` (09/11) — demand by black-box SIMULATION UNDER A REFERENCE POLICY. It replaces
# every simulated set before it (the 05/11 `:v2` with Poisson primary demand, `:v3`,
# `:v4`), removed on 09/11: they replayed the customers against stocks no policy ever
# reaches under the quotas (other items frozen at their current stock, item `i` never
# replenished, every item unlimited at once), and, from 06/11 to 09/11, let only the
# first customer of each epoch buy (a `break` in a two-iterator `for`). See the `:v2`
# section at the end of this file.
#
# The set is process-wide (`set_feature_set!`): the run that trains a model and every
# script that replays it must select the same one. A mismatch is loud — the `Dense`
# layers reject an input of the wrong height. Models trained in the removed sets
# (folders tagged `_fv2` before 09/11, `_fv3`, `_fv4`) can no longer be replayed.
# =============================================================================

const FEATURE_SET = Ref(:v1)

"""
    set_feature_set!(s::Symbol)

Select the feature set (`:v1` or `:v2`) used by [`compute_features`](@ref) for the whole
process. `:v2` needs a reference policy (see [`set_reference_policy!`](@ref)).
"""
function set_feature_set!(s::Symbol)
    s in (:v1, :v2) || throw(ArgumentError(
        "feature set must be :v1 or :v2 (`:v3`, `:v4` and the 05/11 `:v2` were removed on 09/11), got :$s"))
    FEATURE_SET[] = s
    return s
end
feature_set() = FEATURE_SET[]

uses_stock_bound_features(config) = !iszero(over_stock_bound_cost(config))
uses_terminal_features(config) = terminal_stock_penalty(config) > 0

"""
Number of dynamic (state-dependent) columns appended per item.
"""
function nb_dynamic_item_features(config)
    feature_set() === :v1 && return 17
    return 12 + 3 + (uses_terminal_features(config) ? 1 : 0) +
           (uses_stock_bound_features(config) ? 3 : 0)
end

"""
Number of rows of the item block, i.e. the input size of the `θ` model.
"""
item_features_size(config) = feature_count(config) + 1 + nb_dynamic_item_features(config)

"""
Number of stock-level columns appended by [`create_stock_features`](@ref) in the `:v1`
set ([`nb_stock_features`](@ref) gives the count of the active set).
"""
const NB_STOCK_FEATURES = 12

"""
Number of stock-level columns appended per candidate level, in the active set.
"""
function nb_stock_features(config)
    feature_set() === :v1 && return NB_STOCK_FEATURES
    return 8 + 4 + 6 + (uses_terminal_features(config) ? 1 : 0) +
           (uses_stock_bound_features(config) ? 8 : 0)
end

"""
Number of rows of the stock block, i.e. the input size of the `η` model.
Be careful: the feature matrix has one additional row, the item identifier.
"""
stock_features_size(config) = item_features_size(config) + nb_stock_features(config)

"""
$TYPEDSIGNATURES

Create features per item.
The first `feature_count(config) + 1` columns correspond to static features (scaled price
and item features). The remaining [`nb_dynamic_item_features`](@ref) columns are dynamic:

- current *virtual* stock, and scaled with price
- mean sales, and scaled with price
- mean stock, and scaled with price
- mean number of customers in the past
- mean days on lot, and scaled with price
- current *physical* stock, and scaled with price
- stock in transit (`stock - physical_stock`), and scaled with price
- four state-level columns, identical for every item: the total physical stock, the
  slack to `stock_inf` and to `stock_sup`, and the remaining horizon
"""
function create_items_features(state::DRPState)
    config = state.config
    N = item_count(config)
    nb_static = feature_count(config) + 1     # replaces instance.nb_features
    nb_features = item_features_size(config)
    item_features = zeros(Float32, N, nb_features)

    # precompute once
    pos_items = items_with_positive_stock(state)
    mean_sales = mean_sales_history(state)
    mean_stock = mean_stock_history(state)
    current_stock = stock(state)
    phys_stock = physical_stock(state)
    static_features = scaled_features(config)

    # state-level quantities, shared by every item
    total_physical = sum(phys_stock)
    inf_slack = total_physical - stock_inf(config)
    sup_slack = stock_sup(config) - total_physical
    remaining_horizon = max_steps(config) - current_epoch(state)

    for i in 1:N
        p = prices(config)[i]
        ## static features
        item_features[i, 1:nb_static] = static_features[:, i]
        ## current total stock (virtual + physical)
        item_features[i, nb_static + 1] = current_stock[i]
        item_features[i, nb_static + 2] = current_stock[i] * p
        ## mean sales
        ms = item_mean_feature(mean_sales, i, pos_items)
        item_features[i, nb_static + 3] = ms
        item_features[i, nb_static + 4] = ms * p
        ## mean stock
        mst = item_mean_feature(mean_stock, i, pos_items)
        item_features[i, nb_static + 5] = mst
        item_features[i, nb_static + 6] = mst * p
        ## mean customers in the past
        item_features[i, nb_static + 7] = mean_or_zero(state.customer_history)
        ## dol item_features
        dols = compute_dol_item(state, i)
        item_features[i, nb_static + 8] = mean_or_zero(dols)
        item_features[i, nb_static + 9] = mean_or_zero(dols) * p
        ## physical stock
        item_features[i, nb_static + 10] = phys_stock[i]
        item_features[i, nb_static + 11] = phys_stock[i] * p
        ## stock in transit
        in_transit = current_stock[i] - phys_stock[i]
        item_features[i, nb_static + 12] = in_transit
        item_features[i, nb_static + 13] = in_transit * p
        ## state-level features (same for all items)
        item_features[i, nb_static + 14] = total_physical
        item_features[i, nb_static + 15] = inf_slack
        item_features[i, nb_static + 16] = sup_slack
        item_features[i, nb_static + 17] = remaining_horizon
    end
    return item_features
end

"""
$TYPEDSIGNATURES

Create features per stock level per archetype. Row `(i, j)` describes the candidate
post-decision (virtual) stock level `j` for item `i`.

The first `item_features_size(config)` columns repeat the item features, the next
[`NB_STOCK_FEATURES`](@ref) are dynamic stock features, and the last one is the item
identifier used by [`StatisticalModel`](@ref):

- deviation from `stock_inf` and scaled with price
- deviation from `stock_sup` and scaled with price
- deviation from mean stock and scaled with price
- deviation from `max_quotas` and scaled with price
- *coupled* deviation from `stock_inf` and `stock_sup`, i.e. what the total physical stock would be if item
  `i` were at level `j`, and scaled with price
"""
function create_stock_features(state::DRPState, item_features::Matrix{Float32})
    config = state.config
    N = item_count(config)
    ub = ub_per_item(state)
    nb_fi = size(item_features, 2)
    total_rows = sum(ub)
    stock_features = zeros(Float32, total_rows, nb_fi + NB_STOCK_FEATURES + 1) # +1 for unique index
    t = current_epoch(state)

    pos_items = items_with_positive_stock(state)
    mean_stock = mean_stock_history(state)
    phys_stock = physical_stock(state)
    total_physical = sum(phys_stock)

    stock_inf = config.stock_inf
    stock_sup = config.stock_sup

    starts = [1; cumsum(ub)[1:(end - 1)] .+ 1]
    ends = cumsum(ub)

    for i in 1:N
        rows = starts[i]:ends[i]
        stock_features[rows, 1:nb_fi] .= item_features[i:i, :]

        p = prices(config)[i]
        js = 1:ub[i]
        stock_inf_dev = js .- stock_inf
        stock_sup_dev = stock_sup .- js
        stock_mean_dev = js .- item_mean_feature(mean_stock, i, pos_items)
        max_quotas_dev = max_quotas(state.config)[t, i] .- js
        # Total stock if item i ended at level j
        others_physical = total_physical - phys_stock[i]
        total_inf_dev = (others_physical .+ js) .- stock_inf
        total_sup_dev = stock_sup .- (others_physical .+ js)

        stock_features[rows, nb_fi + 1] = stock_inf_dev
        stock_features[rows, nb_fi + 2] = stock_inf_dev .* p
        stock_features[rows, nb_fi + 3] = stock_sup_dev
        stock_features[rows, nb_fi + 4] = stock_sup_dev .* p
        stock_features[rows, nb_fi + 5] = stock_mean_dev
        stock_features[rows, nb_fi + 6] = stock_mean_dev .* p
        stock_features[rows, nb_fi + 7] = max_quotas_dev
        stock_features[rows, nb_fi + 8] = max_quotas_dev .* p
        stock_features[rows, nb_fi + 9] = total_inf_dev
        stock_features[rows, nb_fi + 10] = total_inf_dev .* p
        stock_features[rows, nb_fi + 11] = total_sup_dev
        stock_features[rows, nb_fi + 12] = total_sup_dev .* p
        stock_features[rows, nb_fi + NB_STOCK_FEATURES + 1] .= i # identifier for the item for the statistical model
    end
    return stock_features
end

"""
$TYPEDSIGNATURES

Create features from state, in the active feature set (see [`set_feature_set!`](@ref)).
"""
function compute_features(state::DRPState)
    if feature_set() === :v2
        sim = simulated_reference_rollout(state)
        item_features = create_items_features_v2(state, sim)
        return create_stock_features_v2(state, item_features, sim)'
    end
    # archetype features
    item_features = create_items_features(state)
    # stock features
    stock_features = create_stock_features(state, item_features)
    return stock_features'
end

# =============================================================================
# `:v2` (09/11) — demand by BLACK-BOX simulation, under a REFERENCE POLICY.
#
# Only primitive: customers drawn from the instance's own generator (`generate_scenario`),
# who buy following the environment's purchase rule (the first item they prefer to "no
# purchase" that is in stock — `apply_sales!`) when facing a stock vector. No parameter or
# functional form of the simulator (static utilities, MNL) is read: only counts of
# simulated sales.
#
# Paths: `SIM_PATHS` paths per INSTANCE, drawn once (seeded by the instance only) with the
# preference lists of their customers, kept in a cache. Every state replays epochs t..T of
# these same paths (common random numbers: features differ between states, and between
# the candidate actions of a state, only through the stock and the epoch).
#
# Reference policy: SPGA with κ = remaining horizon (Mahajan & van Ryzin 2001), registered
# by DynamicStoreReplenishment through `set_reference_policy!`. Remarks of 09/11: the
# features must look at stocks a reasonable policy REACHES under the quotas, and they let
# the model do at least as well as the reference (with θ = 0 the level columns describe a
# one-step rollout of SPGA).
#
# Per path `k`, from the current state at `t`:
#
#   trajectory   the reference orders under the quotas at every epoch τ = t..T, then the
#                customers of τ buy: post-decision stocks X^τ, sales V, end stock E.
#   C^τ_i        customers of τ that `i` captures when it is unlimited and every other item
#                is at X^τ. Only read through its tails P̂(C^t ≥ j) and through the
#                censored recursion below — never as a level: an unlimited `i` captures
#                5-25× what it sells (scripts/t5_feature_coherence_v4_v5.jl, 09/11).
#   level j      item `i` at post-decision level j at t, the reference afterwards: at each
#                τ > t, order up to its level S^τ_i within the room the trajectory left to
#                `i` (its own order plus the slack of its quotas); sales min(C^τ_i, ·).
#                For j = 0..ub_i: expected sales over t..T, end stock, and value
#                Q(j) = p·sales − a·end − holding (virtual cost every epoch, physical cost
#                from t + delay on).
#
# Approximations, stated: the other items do not react to `i`'s level (the trajectory is
# shared), and `i`'s sales within an epoch are min(C^τ_i, stock).
#
# Columns:
#   item     P̂(C^t ≥ 1); E[V], E[V]·p (sales of the reference, t..T);
#            a E[E] (end cost on the reference trajectory)
#   level j  P̂(C^t ≥ j), ×p, p P̂(C^t = j − 1), E[min(C^t, j)]  (period t: a unit unsold at
#            t is carried over, hence no `a`);
#            ΔSold(j) = P(the j-th unit sells before T), ×p, m(j) = Q(j) − Q(j − 1),
#            m(j − 1) − m(j) (the η target), Sold(j), j − E[V];
#            a ΔEnd(j) = a P(the j-th unit is left at T)
# Coherence on the test trajectories of SPGA (09/11): Sold(x) explains the realized sales
# t..T with R² 0.97 (n10) and 0.94 (n50).
# =============================================================================

"Number of simulated paths per instance."
const SIM_PATHS = 50

"Unlimited stock for the item whose capture is measured."
const SIM_UNLIMITED = 10_000

"`apply_sales!`'s purchase rule, on a preference list: buy the first item in stock."
function _replay_sales!(sales::Vector{Int}, stock::Vector{Int}, prefs)
    fill!(sales, 0)
    # Two nested loops, NOT `for pref in prefs, j in pref`: in Julia a `break` leaves every
    # iterator of a multi-iterator `for`, so that form let only the first buying customer of
    # each epoch buy (bug of the removed `:v3`/`:v4`, 06/11 to 09/11).
    for pref in prefs
        for j in pref
            if stock[j] > 0
                stock[j] -= 1; sales[j] += 1
                break
            end
        end
    end
    return sales
end

"For each simulated customer, the items he prefers to « no purchase », best first."
function _preference_lists(utilities)
    return map(utilities) do u
        N = length(u) - 1
        order = sortperm(u; rev=true)
        order[1:(findfirst(==(N + 1), order) - 1)]
    end
end

"`P̂(C ≥ j)` for `j = 1..J`."
_sim_tails(c::AbstractVector{Int}, J::Int) = [count(>=(j), c) / length(c) for j in 1:J]

# Preference lists of the customers of every path and epoch: paths[k][τ]
const SIM_PATHS_CACHE = Dict{UInt,Vector{Vector{Vector{Vector{Int}}}}}()
const SIM_PATHS_LOCK = ReentrantLock()

"The `SIM_PATHS` common paths of the instance, drawn at the first call then read."
function simulation_paths(config)
    # `:v4` in the key: the same paths as the removed `:v4` and the 09/11 coherence figure
    key = hash((prices(config), features(config), poisson_arrival_rate(config), max_steps(config), SIM_PATHS, :v4))
    lock(SIM_PATHS_LOCK) do
        get!(SIM_PATHS_CACHE, key) do
            rng = Xoshiro(key)
            map(1:SIM_PATHS) do _
                sc = Utils.generate_scenario(config; rng=rng)
                [_preference_lists(sc.utilities[τ]) for τ in 1:max_steps(config)]
            end
        end
    end
end

const REFERENCE_POLICY = Ref{Any}(nothing)

"""
    set_reference_policy!(f)

Register the reference policy of the `:v2` features. `f(config)` returns
`(; decide, levels)`: `decide(t, x)` is the order (a `Vector{Int}`) at epoch `t` from
virtual stock `x`, `levels(t)` the order-up-to levels at epoch `t`.
DynamicStoreReplenishment registers SPGA (κ = remaining horizon) when it loads.
"""
set_reference_policy!(f) = (REFERENCE_POLICY[] = f)

function reference_policy(config)
    f = REFERENCE_POLICY[]
    isnothing(f) && error("`:v2` features need a reference policy: load " *
                          "DynamicStoreReplenishment (it registers SPGA) or call `set_reference_policy!`.")
    return f(config)
end

"""
    simulated_reference_rollout(state; decided=false) -> NamedTuple

`C1`, `V`, `E` (`K × N`: captures at t, sales and end stock on the reference trajectory)
and, per item, `sold`, `left`, `value` (vectors over the post-decision levels
j = 0..ub_i) — see the `:v2` header.

`decided = true` for a POST-decision state (the SRL critic): the order of epoch `t` is
already in the stock, so the reference does not order at `t`, only from `t + 1` on.
"""
function simulated_reference_rollout(state::DRPState; decided::Bool=false)
    config = state.config
    N = item_count(config)
    t = current_epoch(state); T = max_steps(config); H = T - t + 1
    s = stock(state)
    ref = reference_policy(config)
    paths = simulation_paths(config); K = length(paths)
    A = constraints_matrix(config); Q = quotas(config); mq = max_quotas(config)
    nb_cons = size(A, 1)
    C = zeros(Int, K, N, H); R = zeros(Int, K, N, H)
    V = zeros(Int, K, N); E = zeros(Int, K, N)
    x = similar(s); xx = similar(s); sales = zeros(Int, N); slack = zeros(Int, nb_cons)
    nothing_ordered = zeros(Int, N)
    for k in 1:K
        x .= s
        for h in 1:H
            τ = t + h - 1
            prefs = paths[k][τ]
            # `ref` comes from a `Ref{Any}`: assert, or every use of `y` is dynamic
            y = (decided && h == 1) ? nothing_ordered : ref.decide(τ, x)::Vector{Int}
            x .+= y
            for c in 1:nb_cons
                used = 0
                for i in 1:N
                    used += A[c, i] * y[i]
                end
                slack[c] = Q[τ, c] - used
            end
            for i in 1:N
                r = mq[τ, i] - y[i]
                for c in 1:nb_cons
                    A[c, i] > 0 && (r = min(r, fld(slack[c], A[c, i])))
                end
                R[k, i, h] = y[i] + max(r, 0)
                xx .= x; xx[i] = SIM_UNLIMITED
                C[k, i, h] = _replay_sales!(sales, xx, prefs)[i]
            end
            _replay_sales!(sales, x, prefs)   # x becomes the post-sale stock
            for i in 1:N
                V[k, i] += sales[i]
            end
        end
        E[k, :] .= x
    end

    ub = ub_per_item(state)
    S = Vector{Int}[ref.levels(τ)::Vector{Int} for τ in t:T]
    hv::Vector{Float64} = virtual_stock_cost(config); hp::Vector{Float64} = physical_stock_cost(config)
    L = delivery_delay(config)
    p::Vector{Float64} = prices(config); a = terminal_stock_penalty(config) .* p
    sold = [zeros(ub[i] + 1) for i in 1:N]
    left = [zeros(ub[i] + 1) for i in 1:N]
    value = [zeros(ub[i] + 1) for i in 1:N]
    for i in 1:N, j in 0:ub[i]
        so = 0; le = 0; ho = 0.0
        for k in 1:K
            lev = j
            for h in 1:H
                h > 1 && (lev += clamp(S[h][i] - lev, 0, R[k, i, h]))
                q = min(C[k, i, h], lev); lev -= q; so += q
                # physique dès le pas h = L : l'environnement facture le coût physique de
                # l'époque e sur le stock de e + 1 (`update_cost!`) ; `h > L` jusqu'au 13/11
                ho += (hv[i] + (h >= L ? hp[i] : 0.0)) * lev
            end
            le += lev
        end
        sold[i][j + 1] = so / K; left[i][j + 1] = le / K
        value[i][j + 1] = p[i] * so / K - a[i] * le / K - ho / K
    end
    return (C1=C[:, :, 1], V=V, E=E, sold=sold, left=left, value=value)
end

"""
$TYPEDSIGNATURES

`:v2` item features: the static block, then

- the 12 state columns: virtual stock, mean sales, mean stock (each `× p`), mean
  customers, physical stock, stock in transit (each `× p`), remaining horizon
- `P̂(C^t_i ≥ 1)`; `E[Vᵢ]`, `E[Vᵢ] · pᵢ` (sales of the reference from t to T)
- if `terminal_stock_penalty > 0`: `aᵢ E[Eᵢ]`, the end cost on the reference trajectory
- if `over_stock_bound_cost > 0`: total physical stock, slack to `stock_inf`, `stock_sup`
"""
function create_items_features_v2(state::DRPState, sim)
    config = state.config
    N = item_count(config)
    nb_static = feature_count(config) + 1
    item_features = zeros(Float32, N, item_features_size(config))
    pos_items = items_with_positive_stock(state)
    mean_sales = mean_sales_history(state)
    mean_stock = mean_stock_history(state)
    current_stock = stock(state)
    phys_stock = physical_stock(state)
    static_features = scaled_features(config)
    remaining_horizon = max_steps(config) - current_epoch(state)
    terminal = uses_terminal_features(config)
    bounds = uses_stock_bound_features(config)
    a = terminal_stock_penalty(config) .* prices(config)
    total_physical = sum(phys_stock)
    mean_customers = mean_or_zero(state.customer_history)

    for i in 1:N
        p = prices(config)[i]
        s = current_stock[i]
        c1 = view(sim.C1, :, i)
        col = nb_static
        col!(v) = (col += 1; item_features[i, col] = v)
        item_features[i, 1:nb_static] = static_features[:, i]
        col!(s); col!(s * p)
        ms = item_mean_feature(mean_sales, i, pos_items); col!(ms); col!(ms * p)
        mst = item_mean_feature(mean_stock, i, pos_items); col!(mst); col!(mst * p)
        col!(mean_customers)
        col!(phys_stock[i]); col!(phys_stock[i] * p)
        in_transit = s - phys_stock[i]; col!(in_transit); col!(in_transit * p)
        col!(remaining_horizon)
        col!(count(>=(1), c1) / length(c1))
        v = mean(view(sim.V, :, i)); col!(v); col!(v * p)
        terminal && col!(a[i] * mean(view(sim.E, :, i)))
        if bounds
            col!(total_physical); col!(total_physical - stock_inf(config))
            col!(stock_sup(config) - total_physical)
        end
        @assert col == size(item_features, 2)
    end
    return item_features
end

"""
$TYPEDSIGNATURES

`:v2` stock-level features, row `(i, j)` (candidate post-decision virtual stock `j`),
after the repeated item block:

- `j`, deviation from mean stock, deviation from `max_quotas`, total virtual stock at `j`
  (each `× p`)
- period t: `P̂(C^tᵢ ≥ j)`, `× pᵢ`, `pᵢ P̂(C^tᵢ = j − 1)`, `E[min(C^tᵢ, j)]`
- horizon, level j then the reference: `ΔSold(j)`, `× pᵢ`, `m(j)`, `m(j − 1) − m(j)`,
  `Sold(j)`, `j − E[Vᵢ]`
- if `terminal_stock_penalty > 0`: `aᵢ ΔEnd(j)`, the end cost of the `j`-th unit
- if `over_stock_bound_cost > 0`: the eight `:v1` stock-bound columns
"""
function create_stock_features_v2(state::DRPState, item_features::Matrix{Float32}, sim)
    config = state.config
    N = item_count(config)
    ub = ub_per_item(state)
    nb_fi = size(item_features, 2)
    nb_sf = nb_stock_features(config)
    stock_features = zeros(Float32, sum(ub), nb_fi + nb_sf + 1)
    t = current_epoch(state)
    pos_items = items_with_positive_stock(state)
    mean_stock = mean_stock_history(state)
    phys_stock = physical_stock(state)
    total_physical = sum(phys_stock)
    current_stock = stock(state)
    total_virtual = sum(current_stock)
    terminal = uses_terminal_features(config)
    bounds = uses_stock_bound_features(config)
    α = terminal_stock_penalty(config)

    starts = [1; cumsum(ub)[1:(end - 1)] .+ 1]
    ends = cumsum(ub)
    for i in 1:N
        rows = starts[i]:ends[i]
        stock_features[rows, 1:nb_fi] .= item_features[i:i, :]
        p = prices(config)[i]
        a = α * p
        js = 1:ub[i]
        c1 = view(sim.C1, :, i)
        t1 = _sim_tails(c1, ub[i])
        pmf1 = [count(==(j - 1), c1) / length(c1) for j in js]
        sold = sim.sold[i]; Δsold = diff(sold)
        m = diff(sim.value[i])                      # m[j] = Q(j) − Q(j − 1)
        col = nb_fi
        col!(v) = (col += 1; stock_features[rows, col] = v)
        col!(js); col!(js .* p)
        dev = js .- item_mean_feature(mean_stock, i, pos_items); col!(dev); col!(dev .* p)
        q = max_quotas(config)[t, i] .- js; col!(q); col!(q .* p)
        tot = (total_virtual - current_stock[i]) .+ js; col!(tot); col!(tot .* p)
        col!(t1); col!(t1 .* p); col!(p .* pmf1); col!(cumsum(t1))
        col!(Δsold); col!(Δsold .* p); col!(m)
        col!([0.0; m[1:(end - 1)] .- m[2:end]])     # m(j − 1) − m(j), the η target
        col!(sold[js .+ 1]); col!(js .- mean(view(sim.V, :, i)))
        terminal && col!(a .* diff(sim.left[i]))
        if bounds
            others_physical = total_physical - phys_stock[i]
            inf_dev = js .- stock_inf(config); col!(inf_dev); col!(inf_dev .* p)
            sup_dev = stock_sup(config) .- js; col!(sup_dev); col!(sup_dev .* p)
            tinf = (others_physical .+ js) .- stock_inf(config); col!(tinf); col!(tinf .* p)
            tsup = stock_sup(config) .- (others_physical .+ js); col!(tsup); col!(tsup .* p)
        end
        @assert col == nb_fi + nb_sf
        stock_features[rows, nb_fi + nb_sf + 1] .= i
    end
    return stock_features
end
