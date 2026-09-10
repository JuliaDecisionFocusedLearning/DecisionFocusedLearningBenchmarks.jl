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

"""
Number of dynamic (state-dependent) columns appended per item.
"""
nb_dynamic_item_features(config) = 17

"""
Number of rows of the item block, i.e. the input size of the `θ` model.
"""
item_features_size(config) = feature_count(config) + 1 + nb_dynamic_item_features(config)

"""
Number of stock-level columns appended by [`create_stock_features`](@ref).
"""
const NB_STOCK_FEATURES = 12

"""
Number of rows of the stock block, i.e. the input size of the `η` model.
Be careful: the feature matrix has one additional row, the item identifier.
"""
stock_features_size(config) = item_features_size(config) + NB_STOCK_FEATURES

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

Create features from state.
"""
function compute_features(state::DRPState)
    # archetype features
    item_features = create_items_features(state)
    # stock features
    stock_features = create_stock_features(state, item_features)
    return stock_features'
end