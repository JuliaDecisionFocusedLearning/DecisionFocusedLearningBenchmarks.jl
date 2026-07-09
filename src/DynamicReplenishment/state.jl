"""
$TYPEDSIGNATURES

State data structure for the Dynamic Replenishment Problem.
Convention: all history matrices are (time, item), i.e. `history[t, i]`.
"""
@kwdef mutable struct DRPState{B<:DynamicReplenishmentBenchmark}
    config::B
    current_epoch::Int
    stock::Vector{Int}
    stock_history::Matrix{Int}          # (current_epoch, N)
    replenishment_history::Matrix{Int}  # (current_epoch-1,   N)
    sales_history::Matrix{Int}          # (current_epoch-1,   N)
    customer_history::Vector{Int}       # (current_epoch -1)
    ub_per_item::Vector{Int}
    current_cost::Float64 = 0.0
end

function DRPState{B}(
    config::B, stock_ini::Vector{Int}
) where {B<:DynamicReplenishmentBenchmark}
    N = length(stock_ini)
    return DRPState{B}(;
        config,
        current_epoch=1,
        stock=copy(stock_ini),
        stock_history=reshape(copy(stock_ini), 1, N),
        replenishment_history=zeros(Int, 0, N),
        sales_history=zeros(Int, 0, N),
        customer_history=Int[],
        ub_per_item=stock_ini .+ max_quotas(config)[1],
        current_cost=0.0,
    )
end

function DRPState(
    config::B, stock_ini::Vector{Int}
) where {B<:DynamicReplenishmentBenchmark}
    return DRPState{B}(config, stock_ini)
end

function total_sales_per_epoch(state::DRPState)
    return vec(sum(state.sales_history; dims=2))
end

current_epoch(state::DRPState) = state.current_epoch
stock(state::DRPState) = state.stock
total_stock(state::DRPState) = sum(state.stock)
stock_history(state::DRPState) = state.stock_history
replenishment_history(state::DRPState) = state.replenishment_history
sales_history(state::DRPState) = state.sales_history
customer_history(state::DRPState) = state.customer_history
stock_ini(state::DRPState) = stock_history(state)[1, :]
current_cost(state::DRPState) = state.current_cost
ub_per_item(state::DRPState) = state.ub_per_item

function reset_state!(state::DRPState)
    N = item_count(state.config)
    s0 = stock_ini(state)
    state.current_epoch = 1
    # TODO: mettre un nouveau stock initial 
    state.stock = copy(s0)
    state.stock_history = reshape(copy(s0), 1, N)
    state.replenishment_history = zeros(Int, 0, N)
    state.sales_history = zeros(Int, 0, N)
    state.customer_history = Int[]
    state.ub_per_item = s0 .+ max_quotas(state.config)[1]
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

function physical_stock(state::DRPState, t::Int)
    config = state.config
    s0 = stock_ini(state)
    N = item_count(config)
    t ≤ delivery_delay(config) && return zeros(Int, N)
    t_repl = t - delivery_delay(config)      # replenishments received by time t
    t_sales = t - 1                          # sales completed by time t
    repl_sum = vec(sum(view(replenishment_history(state), 1:t_repl, :); dims=1))
    sales_sum = if t_sales == 0
        zeros(Int, N)
    else
        vec(sum(view(sales_history(state), 1:t_sales, :); dims=1))
    end
    return max.(0, s0 .+ repl_sum .- sales_sum)
end

function current_physical_stock(state::DRPState)
    return physical_stock(state, current_epoch(state) + 1)
end

function update_cost!(state::DRPState)
    config = state.config
    t = current_epoch(state)
    # sales reward
    sales_t = view(state.sales_history, t, :)
    margin = sum(prices(config) .* sales_t)
    # physical stock cost
    phys_stock = current_physical_stock(state)
    physical_cost = sum(physical_stock_cost(config) .* phys_stock)
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

function compute_cost(
    state::DRPState, next_replenishment::Vector{Int}, next_sales::Vector{Int}
)
    total = 0.0
    config = state.config
    replenishments = vcat(replenishment_history(state), next_replenishment')
    sales = vcat(sales_history(state), next_sales')
    stock_hist = vcat(
        state.stock_history, (stock(state) .+ next_replenishment .- next_sales)'
    )
    state_ = DRPState(;
        config=config,
        current_epoch=current_epoch(state) + 1,
        stock=stock_hist[end, :],
        stock_history=stock_hist,
        replenishment_history=replenishments,
        sales_history=sales,
        customer_history=customer_history(state),
        ub_per_item=stock_hist[end, :] .+ max_quotas(config)[current_epoch(state), :],
        current_cost=0.0,
    )
    for t in 1:current_epoch(state)
        # margin
        sales_t = view(sales, t, :)
        total += sum(prices(config) .* sales_t)
        # virtual stock cost
        virtual_stock = stock_hist[t + 1, :]
        total -= sum(virtual_stock_cost(config) .* virtual_stock)
        # physical stock cost
        phys_stock = physical_stock(state_, t + 1)
        total -= sum(physical_stock_cost(config) .* phys_stock)
        # over / under stock costs
        s = sum(virtual_stock)
        total -=
            over_stock_bound_cost(config) *
            (max(0, stock_inf(config) - s) + max(0, s - stock_sup(config)))
    end
    return total
end

function apply_replenishment!(state::DRPState, replenishment::Vector{Int})
    state.stock .+= replenishment
    return state.replenishment_history = vcat(state.replenishment_history, replenishment')
end

function apply_sales!(
    state::DRPState; nb_customers::Int, utilities::Vector{Vector{Float64}}
)
    N = length(state.stock)
    sales = zeros(Int, N)
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
    return delta_cost
end

function add_customers!(
    state::DRPState; nb_customers::Int, utilities::Vector{Vector{Float64}}
)
    state.customer_history = push!(state.customer_history, nb_customers)
    return state
end
