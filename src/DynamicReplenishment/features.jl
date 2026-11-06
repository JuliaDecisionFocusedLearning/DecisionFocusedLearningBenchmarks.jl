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
# `:v1` (default) is the historical set, unchanged. `:v2` (2026-11-05) drops the
# days-on-lot columns, keeps the stock-bound columns only when they cost something
# (`over_stock_bound_cost > 0`), and adds primary-demand columns estimated by
# simulating the instance's customer generator (no policy needed: every item
# available, first choice counted), crossed with the remaining horizon — plus the
# expected end-of-horizon cost when `terminal_stock_penalty > 0`. See
# [`create_items_features_v2`](@ref) and [`create_stock_features_v2`](@ref).
#
# The set is process-wide (`set_feature_set!`): the run that trains a model and every
# script that replays it must select the same one. A mismatch is loud — the `Dense`
# layers reject an input of the wrong height.
# =============================================================================

const FEATURE_SET = Ref(:v1)

"""
    set_feature_set!(s::Symbol)

Select the feature set (`:v1` or `:v2`) used by [`compute_features`](@ref) for the
whole process.
"""
function set_feature_set!(s::Symbol)
    s in (:v1, :v2, :v3) || throw(ArgumentError("feature set must be :v1, :v2 or :v3, got :$s"))
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
    feature_set() === :v3 && return 12 + 3 + (uses_terminal_features(config) ? 1 : 0) +
                                    (uses_stock_bound_features(config) ? 3 : 0)
    return 12 + 4 + (uses_terminal_features(config) ? 1 : 0) +
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
    feature_set() === :v3 && return 8 + 4 + (uses_terminal_features(config) ? 2 : 0) +
                                    (uses_stock_bound_features(config) ? 8 : 0)
    return 8 + 3 + (uses_terminal_features(config) ? 1 : 0) +
           (uses_stock_bound_features(config) ? 8 : 0)
end

"""
Number of rows of the stock block, i.e. the input size of the `η` model.
Be careful: the feature matrix has one additional row, the item identifier.
"""
stock_features_size(config) = item_features_size(config) + nb_stock_features(config)

# --- primary demand, by simulation -----------------------------------------------

const PRIMARY_DEMAND_CACHE = Dict{UInt,Vector{Float64}}()
const PRIMARY_DEMAND_LOCK = ReentrantLock()

"""
    primary_demand(config; nb_scenarios=200)

Expected number of customers per period whose FIRST choice is item `i`, every item
being available: the primary demand `μᵢ`. Estimated by simulating `nb_scenarios`
scenarios of the instance's own generator (`generate_scenario`), i.e. the same
estimator as the OR baselines (`estimate_primary_demand` in DynamicStoreReplenishment):
no policy is involved. Computed once per instance (content hash of the static
utilities and arrival rate) and cached, thread-safe; the rng is seeded from that hash.
"""
function primary_demand(config; nb_scenarios::Int=200)
    key = hash((config.static_utilities, poisson_arrival_rate(config), nb_scenarios))
    cached = lock(() -> get(PRIMARY_DEMAND_CACHE, key, nothing), PRIMARY_DEMAND_LOCK)
    isnothing(cached) || return cached
    N = item_count(config)
    rng = Xoshiro(key)
    counts = zeros(Float64, N)
    periods = 0
    for _ in 1:nb_scenarios
        scenario = Utils.generate_scenario(config; rng=rng)
        for utilities_t in scenario.utilities
            periods += 1
            for u in utilities_t
                c = argmax(u)
                c <= N && (counts[c] += 1.0)
            end
        end
    end
    μ = periods == 0 ? counts : counts ./ periods
    return lock(() -> get!(PRIMARY_DEMAND_CACHE, key, μ), PRIMARY_DEMAND_LOCK)
end

"`P(D ≥ j)` for `D ~ Poisson(m)`."
poisson_tail(m::Real, j::Integer) = j <= 0 ? 1.0 : ccdf(Poisson(m), j - 1)

"`E[(s − D)⁺]` for `D ~ Poisson(m)`: expected unsold units out of `s`."
expected_leftover(m::Real, s::Integer) = s <= 0 ? 0.0 : sum(cdf(Poisson(m), j - 1) for j in 1:s)

"Remaining periods INCLUDING the current one, `τ = T − t + 1`."
remaining_periods(state::DRPState) = max_steps(state.config) - current_epoch(state) + 1

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

`:v2` item features. Same static block as `:v1`; dynamic columns, per item `i`, with
`τ = T − t + 1` the remaining periods, `sᵢ` the current virtual stock, `μᵢ` the primary
demand per period ([`primary_demand`](@ref)) and `Dᵢ ~ Poisson(τ μᵢ)` the primary demand
left until the end:

- current virtual stock, and scaled with price
- mean sales, and scaled with price
- mean stock, and scaled with price
- mean number of customers in the past
- current physical stock, and scaled with price
- stock in transit, and scaled with price
- remaining horizon `T − t`
- `μᵢ`, `μᵢ · pᵢ`, `τ μᵢ`, `P(Dᵢ ≥ 1)`: the SCALE of the demand, nothing that looks
  like a decision rule — the probability that the next unit sells, `P(Dᵢ ≥ sᵢ + 1)`, is
  deliberately left to the level block (`P(Dᵢ ≥ j)` at `j = sᵢ + 1`), where `η` reads it
- if `terminal_stock_penalty > 0`, with `aᵢ = α pᵢ` the end-of-horizon cost of a unit:
  `aᵢ E[(sᵢ − Dᵢ)⁺]`, the expected end cost of the stock already there (a state
  feature: the level block does not carry it)
- if `over_stock_bound_cost > 0`: total physical stock, slack to `stock_inf` and to
  `stock_sup`

No days on lot: on short episodes it mostly equals its no-sale fallback, the absolute
epoch, i.e. a disguised time feature.
"""
function create_items_features_v2(state::DRPState)
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
    μ = primary_demand(config)
    τ = remaining_periods(state)
    remaining_horizon = max_steps(config) - current_epoch(state)
    terminal = uses_terminal_features(config)
    bounds = uses_stock_bound_features(config)
    a = terminal_stock_penalty(config) .* prices(config)
    total_physical = sum(phys_stock)
    mean_customers = mean_or_zero(state.customer_history)

    for i in 1:N
        p = prices(config)[i]
        s = current_stock[i]
        m = τ * μ[i]
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
        col!(μ[i]); col!(μ[i] * p); col!(m); col!(poisson_tail(m, 1))
        if terminal
            col!(a[i] * expected_leftover(m, s))
        end
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

`:v2` stock-level features. Row `(i, j)` describes the candidate post-decision virtual
stock level `j`; after the repeated item block come, with `Dᵢ ~ Poisson(τ μᵢ)`:

- `j`, and scaled with price (the level itself: in `:v1` it only came in through the
  stock-bound deviations, `j − stock_inf` with `stock_inf = 0`)
- deviation from mean stock, and scaled with price
- deviation from `max_quotas`, and scaled with price
- total virtual stock if item `i` were at level `j`, and scaled with price (the coupling
  between items, without the bound semantics)
- `P(Dᵢ ≥ j)` (the `j`-th unit sells before the end), and scaled with price
- `j − τ μᵢ` (expected excess over the remaining primary demand)
- if `terminal_stock_penalty > 0`: `α pᵢ (1 − P(Dᵢ ≥ j))`, expected end cost of the `j`-th unit
- if `over_stock_bound_cost > 0`: the eight `:v1` stock-bound columns
"""
function create_stock_features_v2(state::DRPState, item_features::Matrix{Float32})
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
    μ = primary_demand(config)
    τ = remaining_periods(state)
    terminal = uses_terminal_features(config)
    bounds = uses_stock_bound_features(config)
    α = terminal_stock_penalty(config)

    starts = [1; cumsum(ub)[1:(end - 1)] .+ 1]
    ends = cumsum(ub)
    for i in 1:N
        rows = starts[i]:ends[i]
        stock_features[rows, 1:nb_fi] .= item_features[i:i, :]
        p = prices(config)[i]
        js = 1:ub[i]
        m = τ * μ[i]
        tails = [poisson_tail(m, j) for j in js]
        col = nb_fi
        col!(v) = (col += 1; stock_features[rows, col] = v)
        col!(js); col!(js .* p)
        dev = js .- item_mean_feature(mean_stock, i, pos_items); col!(dev); col!(dev .* p)
        q = max_quotas(config)[t, i] .- js; col!(q); col!(q .* p)
        tot = (total_virtual - current_stock[i]) .+ js; col!(tot); col!(tot .* p)
        col!(tails); col!(tails .* p); col!(js .- m)
        terminal && col!(α * p .* (1 .- tails))
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

"""
$TYPEDSIGNATURES

Create features from state, in the active feature set (see [`set_feature_set!`](@ref)).
"""
function compute_features(state::DRPState)
    if feature_set() === :v3
        C = simulated_captures(state)
        item_features = create_items_features_v3(state, C)
        return create_stock_features_v3(state, item_features, C)'
    end
    if feature_set() === :v2
        item_features = create_items_features_v2(state)
        return create_stock_features_v2(state, item_features)'
    end
    # archetype features
    item_features = create_items_features(state)
    # stock features
    stock_features = create_stock_features(state, item_features)
    return stock_features'
end
# =============================================================================
# `:v3` (2026-11-06) — demand by BLACK-BOX simulation, substitution included.
#
# The only primitive: customers drawn from the instance's own generator
# (`generate_scenario`), who buy following the environment's purchase rule (the
# first item they prefer to "no purchase" that is in stock — `apply_sales!`) when
# facing a stock vector. No parameter or functional form of the simulator (static
# utilities, MNL) is read: only counts of simulated sales.
#
# For a state (virtual stock `s` before the decision, epoch `t`) and K paths of the
# customers of epoch `t` ONLY (one period: no replenishment can happen inside it, so
# no policy is needed to play the path):
#
#     C_i(ω) = sales of item i on path ω when i has unlimited stock and every other
#              item k keeps its current stock s_k,
#
# i.e. the customers item i would capture this period, substitution included (a
# customer whose favourite is out of stock and who falls back on i is counted).
# Diagnostic (scripts/t5_feature_simulation_diagnostic.jl, n50 α=0.5): E[min(C_i, x_i)]
# explains R² ≈ 0.41 of the realized sales, against 0.10 for the primary demand μᵢ.
# =============================================================================

"Number of simulated paths per state for the `:v3` features."
const SIM_PATHS = 100

"Unlimited stock for the item whose capture is measured."
const SIM_UNLIMITED = 10_000

"`apply_sales!`'s purchase rule, on a preference list: buy the first item in stock."
function _replay_sales!(sales::Vector{Int}, stock::Vector{Int}, prefs)
    fill!(sales, 0)
    for pref in prefs, j in pref
        if stock[j] > 0
            stock[j] -= 1; sales[j] += 1
            break
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

"""
    simulated_captures(state; nb_paths=SIM_PATHS) -> Matrix{Int} (nb_paths × N)

`C[k, i]` = customers of the current epoch that item `i` captures on simulated path `k`
when it has unlimited stock and every other item keeps its current (virtual) stock.
The rng is seeded from the instance and the state: a given state always gets the same
features (thread-safe, no shared cache).
"""
function simulated_captures(state::DRPState; nb_paths::Int=SIM_PATHS)
    config = state.config
    N = item_count(config)
    t = current_epoch(state)
    s = stock(state)
    rng = Xoshiro(hash((prices(config), features(config), poisson_arrival_rate(config), t, s, :v3)))
    C = zeros(Int, nb_paths, N)
    sales = zeros(Int, N); x = similar(s)
    for k in 1:nb_paths
        prefs = _preference_lists(Utils.generate_scenario(config; rng=rng).utilities[t])
        for i in 1:N
            x .= s; x[i] = SIM_UNLIMITED
            C[k, i] = _replay_sales!(sales, x, prefs)[i]
        end
    end
    return C
end

"`P̂(C_i ≥ j)` for `j = 1..J`."
_sim_tails(c::AbstractVector{Int}, J::Int) = [count(>=(j), c) / length(c) for j in 1:J]

"""
$TYPEDSIGNATURES

`:v3` item features: the `:v2` block with the Poisson primary-demand columns replaced by
simulated ones (see [`simulated_captures`](@ref)):

- the 12 state columns of `:v2` (stocks, mean sales / stock / customers, physical stock,
  in transit, remaining horizon `T − t`)
- `d̂ᵢ = E[Cᵢ]`, `d̂ᵢ · pᵢ`, `P̂(Cᵢ ≥ 1)`
- if `terminal_stock_penalty > 0`: `aᵢ E[(sᵢ − Cᵢ)⁺]`, the expected unsold part of the
  stock already there after this period, at the end-of-horizon cost `aᵢ = α pᵢ`
- if `over_stock_bound_cost > 0`: total physical stock, slack to `stock_inf`, `stock_sup`
"""
function create_items_features_v3(state::DRPState, C::AbstractMatrix{Int})
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
        c = view(C, :, i)
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
        d = mean(c); col!(d); col!(d * p); col!(count(>=(1), c) / length(c))
        terminal && col!(a[i] * mean(max.(s .- c, 0)))
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

`:v3` stock-level features. Row `(i, j)` describes the candidate post-decision level `j`;
after the repeated item block:

- the 8 `:v2` columns: `j`, deviation from mean stock, from `max_quotas`, total virtual
  stock at level `j` (each with its `× p` twin)
- `P̂(Cᵢ ≥ j)` (the `j`-th unit sells this period), and `× pᵢ`
- `(pᵢ + aᵢ) P̂(Cᵢ = j − 1)`: the INCREMENT of the marginal value between units `j − 1`
  and `j` — what `η[i][j]` produces in `m_i(j) = θ_i − Σ_{k≤j} η[i][k]`, and which a
  row-wise model cannot rebuild from the level `P̂(Cᵢ ≥ j)` alone (diagnostic: R² 0.25)
- `E[min(Cᵢ, j)]`, the expected sales of `i` at level `j`
- if `terminal_stock_penalty > 0`: `aᵢ (1 − P̂(Cᵢ ≥ j))`, the end cost of the `j`-th unit
  if this period were the last, and the same × `1{t = T}` (a linear model cannot make the
  weight of a column depend on `t` by itself)
- if `over_stock_bound_cost > 0`: the eight `:v1` stock-bound columns
"""
function create_stock_features_v3(state::DRPState, item_features::Matrix{Float32}, C::AbstractMatrix{Int})
    config = state.config
    N = item_count(config)
    ub = ub_per_item(state)
    nb_fi = size(item_features, 2)
    nb_sf = nb_stock_features(config)
    stock_features = zeros(Float32, sum(ub), nb_fi + nb_sf + 1)
    t = current_epoch(state)
    last = t == max_steps(config) ? 1.0 : 0.0
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
        c = view(C, :, i)
        tails = _sim_tails(c, ub[i])
        pmf = [count(==(j - 1), c) / length(c) for j in js]
        col = nb_fi
        col!(v) = (col += 1; stock_features[rows, col] = v)
        col!(js); col!(js .* p)
        dev = js .- item_mean_feature(mean_stock, i, pos_items); col!(dev); col!(dev .* p)
        q = max_quotas(config)[t, i] .- js; col!(q); col!(q .* p)
        tot = (total_virtual - current_stock[i]) .+ js; col!(tot); col!(tot .* p)
        col!(tails); col!(tails .* p)
        col!((p + a) .* pmf)
        col!(cumsum(tails))
        if terminal
            col!(a .* (1 .- tails)); col!(last .* a .* (1 .- tails))
        end
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
