"""
$TYPEDEF

State data structure for the Dynamic Replenishment Problem.
Convention: all history matrices are (time, item), i.e. `history[t, i]`.

# Fields
$TYPEDFIELDS
"""
mutable struct DRPState{B<:DynamicReplenishmentBenchmark}
    "The benchmark configuration."
    config::B
    "Current time step (epoch) of the simulation."
    current_epoch::Int
    "Current stock levels for each item (N)"
    stock::Vector{Int}
    "Current physical stock levels for each item (N)"
    physical_stock::Vector{Int}
    "History of stock levels for each item (current_epoch, N)"
    stock_history::Matrix{Int}
    "History of replenishments for each item (current_epoch-1, N)"
    replenishment_history::Matrix{Int}
    "History of sales for each item (current_epoch-1, N)"
    sales_history::Matrix{Int}
    "History of number of customers per time step (current_epoch-1)"
    customer_history::Vector{Int}
    "Upper bound of replenishment per item (N)"
    ub_per_item::Vector{Int}
    "Current cost of the state"
    current_cost::Float64
end

"""
$TYPEDSIGNATURES

Compute physical stock at time `t` from raw historical data.
"""
function compute_physical_stock(
    config,
    t::Int,
    s0::Vector{Int},
    replenishment_history::AbstractMatrix{Int},
    sales_history::AbstractMatrix{Int},
)
    N = item_count(config)
    t_repl = t - delivery_delay(config)
    t_sales = t - 1
    sales_sum = if t_sales <= 0
        zeros(Int, N)
    else
        vec(sum(view(sales_history, 1:t_sales, :); dims=1))
    end
    if t <= delivery_delay(config)
        return max.(0, s0 .- sales_sum)
    else
        repl_sum = if t_repl <= 0
            zeros(Int, N)
        else
            vec(sum(view(replenishment_history, 1:t_repl, :); dims=1))
        end
        return max.(0, s0 .+ repl_sum .- sales_sum)
    end
end

"""
$TYPEDSIGNATURES

Compute the cumulative cost of a complete history (epochs 1:T where T = size(sales_history,1)),
from raw data.
"""
function compute_total_cost(
    config,
    s0::Vector{Int},
    stock_history::AbstractMatrix{Int},
    replenishment_history::AbstractMatrix{Int},
    sales_history::AbstractMatrix{Int},
)
    T = size(sales_history, 1)
    T == 0 && return 0.0

    margin_sales = sum(sum(prices(config) .* sales_history[t, :]) for t in 1:T)
    v_stock_cost = sum(
        sum(virtual_stock_cost(config) .* stock_history[t + 1, :] for t in 1:T)
    )
    phys_stocks = [
        compute_physical_stock(config, t + 1, s0, replenishment_history, sales_history) for
        t in 1:T
    ]
    p_stock_cost = sum(sum(physical_stock_cost(config) .* phys_stocks[t]) for t in 1:T)
    stock_bound_cost = sum(
        over_stock_bound_cost(config) * (
            max(0, stock_inf(config) - sum(phys_stocks[t])) +
            max(0, sum(phys_stocks[t]) - stock_sup(config))
        ) for t in 1:T
    )
    return margin_sales - v_stock_cost - p_stock_cost - stock_bound_cost
end

"""
$TYPEDSIGNATURES

Construct a `DRPState`. 
By convention, the initial physical stock is equal to the initial stock.
"""

function DRPState(;
    config::B,
    current_epoch::Int,
    stock::Vector{Int},
    stock_history::Matrix{Int},
    replenishment_history::Matrix{Int},
    sales_history::Matrix{Int},
    customer_history::Vector{Int},
    ub_per_item::Vector{Int},
) where {B<:DynamicReplenishmentBenchmark}
    s0 = stock_history[1, :]
    physical_stock = compute_physical_stock(
        config, current_epoch, s0, replenishment_history, sales_history
    )
    current_cost = compute_total_cost(
        config, s0, stock_history, replenishment_history, sales_history
    )
    return DRPState{B}(
        config,
        current_epoch,
        stock,
        physical_stock,
        stock_history,
        replenishment_history,
        sales_history,
        customer_history,
        ub_per_item,
        current_cost,
    )
end

function DRPState(
    config::B, stock_ini::Vector{Int}
) where {B<:DynamicReplenishmentBenchmark}
    N = length(stock_ini)
    return DRPState(;
        config,
        current_epoch=1,
        stock=copy(stock_ini),
        stock_history=reshape(copy(stock_ini), 1, N),
        replenishment_history=zeros(Int, 0, N),
        sales_history=zeros(Int, 0, N),
        customer_history=Int[],
        ub_per_item=stock_ini .+ max_quotas(config)[1, :],
    )
end

current_epoch(state::DRPState) = state.current_epoch
stock(state::DRPState) = state.stock
physical_stock(state::DRPState) = state.physical_stock
total_stock(state::DRPState) = sum(state.stock)
stock_history(state::DRPState) = state.stock_history
replenishment_history(state::DRPState) = state.replenishment_history
sales_history(state::DRPState) = state.sales_history
customer_history(state::DRPState) = state.customer_history
stock_ini(state::DRPState) = stock_history(state)[1, :]
current_cost(state::DRPState) = state.current_cost
ub_per_item(state::DRPState) = state.ub_per_item

function reset_state!(state::DRPState, rng::AbstractRNG; reset_stock_ini=false)
    N = item_count(state.config)
    if reset_stock_ini
        s0 = rand(rng, 0:10, N)
    else
        s0 = stock_ini(state)
    end
    state.current_epoch = 1
    state.stock = copy(s0)
    state.physical_stock = copy(s0)
    state.stock_history = reshape(copy(s0), 1, N)
    state.replenishment_history = zeros(Int, 0, N)
    state.sales_history = zeros(Int, 0, N)
    state.customer_history = Int[]
    state.ub_per_item = s0 .+ max_quotas(state.config)[1, :]
    state.current_cost = 0.0
    return state
end

function is_feasible(state::DRPState, replenishment::Vector{Int}; verbose=false)
    config = state.config
    cons_mat = constraints_matrix(config)
    q = quotas(config)
    for c in 1:nb_constraints(config)
        if sum(cons_mat[c, :] .* replenishment) > q[state.current_epoch, c]
            verbose &&
                @warn "Replenishment violates quota constraint $c at epoch $(state.current_epoch) : $(sum(cons_mat[c, :] .* replenishment)) > $(q[state.current_epoch, c])"
            return false
        end
    end
    return true
end

function update_cost!(state::DRPState)
    config = state.config
    t = current_epoch(state)
    # sales reward
    sales_t = view(state.sales_history, t, :)
    margin = sum(prices(config) .* sales_t)
    # physical stock cost
    physical_cost = sum(physical_stock_cost(config) .* state.physical_stock)
    # virtual stock cost
    virtual_stock = view(state.stock_history, t + 1, :)
    virtual_cost = sum(virtual_stock_cost(config) .* virtual_stock)
    # over / under stock costs
    total = sum(virtual_stock)
    under = max(0, stock_inf(config) - total)
    over = max(0, total - stock_sup(config))
    penalty = over_stock_bound_cost(config) * (under + over)

    delta = margin - virtual_cost - physical_cost - penalty
    state.current_cost += delta
    return delta
end

function apply_replenishment!(state::DRPState, replenishment::Vector{Int})
    state.stock .+= replenishment
    state.replenishment_history = vcat(state.replenishment_history, replenishment')
    return nothing
end

function apply_sales!(state::DRPState; utilities::Vector{Vector{Float64}})
    N = length(state.stock)
    sales = zeros(Int, N)
    nb_customers = length(utilities)
    for k in 1:nb_customers
        order_of_sales = sortperm(utilities[k]; rev=true)
        for item_index in order_of_sales
            if item_index == N + 1
                break
            end
            if state.stock[item_index] > 0
                sales[item_index] += 1
                state.stock[item_index] -= 1
                break
            end
        end
    end
    state.sales_history = vcat(state.sales_history, sales')
    state.stock_history = vcat(state.stock_history, state.stock')
    delta_cost = update_cost!(state)
    state.physical_stock = compute_physical_stock(
        state.config,
        current_epoch(state),
        stock_ini(state),
        state.replenishment_history,
        state.sales_history,
    )
    return delta_cost
end

function add_customers!(state::DRPState; utilities::Vector{Vector{Float64}})
    nb_customers = length(utilities)
    state.customer_history = push!(state.customer_history, nb_customers)
    return state
end
