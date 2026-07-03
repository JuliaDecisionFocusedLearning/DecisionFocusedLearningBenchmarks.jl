function mean_feature_matrix(state::DRPState, feature_matrix)
    N = item_count(state.config)
    @assert size(feature_matrix, 2) == N
    isempty(feature_matrix) && return zeros(Float64, N)
    return vec(mean(feature_matrix; dims=1))
end

mean_sales_history(state::DRPState) = mean_feature_matrix(state, sales_history(state))
function mean_replenishment_history(state::DRPState)
    mean_feature_matrix(state, replenishment_history(state))
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
            findfirst(>=(j), cum_sales)
        else
            t_now
        end

        dols[j] = end_date_j - date_repl_j + 1
    end
    return dols
end

"""
$TYPEDSIGNATURES

Create features per item. 
The first nb_features columns correspond to static features (price + dols).
The last 6 columns correspond to dynamic features:
- current stock and scaled with price
- mean sales and scaled with price
- mean days on lot and scaled with price (to be implemented)
"""
function create_items_features(state::DRPState)
    config = state.config
    N = item_count(config)
    nb_static = feature_count(config) + 1     # replaces instance.nb_features
    nb_features = nb_static + 9
    item_features = zeros(N, nb_features)

    # precompute once
    pos_items = items_with_positive_stock(state)
    mean_sales = mean_sales_history(state)
    mean_stock = mean_stock_history(state)
    current_stock = stock(state)
    static_features = vcat(reshape(prices(config), 1, :), features(config))

    for i in 1:N
        p = prices(config)[i]
        ## static features
        item_features[i, 1:nb_static] = static_features[:, i]
        ## current stock
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
        ## diol item_features
        dols = compute_dol_item(state, i)
        item_features[i, nb_static + 8] = mean_or_zero(dols)
        item_features[i, nb_static + 9] = mean_or_zero(dols) * p
    end
    return item_features
end

"""
$TYPEDSIGNATURES

Create features per stock level per archetype.
The first instance.nb_features+6 columns correspond to static the archetype features.
The last 8 columns correspond to dynamic stock features:
- deviation from stock_inf and scaled with price 
- deviation from stock_sup and scaled with price
- deviation from min_quota and i and scaled with price
- deviation from mean stock and scaled with price
"""
function create_stock_features(state::DRPState, item_features::Matrix{Float64})
    config = state.config
    N = item_count(config)
    ub = ub_per_item(state)
    nb_fi = size(item_features, 2)
    total_rows = sum(ub)
    stock_features = zeros(total_rows, nb_fi + 8)
    t = current_epoch(state)

    pos_items = items_with_positive_stock(state)
    mean_stock = mean_stock_history(state)

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

        stock_features[rows, nb_fi + 1] = stock_inf_dev
        stock_features[rows, nb_fi + 2] = stock_inf_dev .* p
        stock_features[rows, nb_fi + 3] = stock_sup_dev
        stock_features[rows, nb_fi + 4] = stock_sup_dev .* p
        stock_features[rows, nb_fi + 5] = stock_mean_dev
        stock_features[rows, nb_fi + 6] = stock_mean_dev .* p
        stock_features[rows, nb_fi + 7] = max_quotas_dev
        stock_features[rows, nb_fi + 8] = max_quotas_dev .* p
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
    normalize_features!(stock_features)
    # normalize_features!(item_features)
    return stock_features'
end
