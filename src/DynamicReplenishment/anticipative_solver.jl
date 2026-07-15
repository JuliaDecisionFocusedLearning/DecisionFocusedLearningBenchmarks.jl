
"""
$TYPEDSIGNATURES

Compute big M values for a scenario of a specific environment.
"""
function compute_bigM!(env::Environment, scenario::Scenario)
    T = max_steps(env.config)
    N = item_count(env.config)
    max_q = max_quotas(env.config)
    s0 = stock(env)
    n_customers = nb_customers(scenario)
    utilities = scenario.utilities
    big_M = Vector{Vector{Vector{Int}}}(undef, T - current_epoch(env) + 1)
    for (t_m, t) in enumerate(current_epoch(env):T)
        big_M[t_m] = Vector{Vector{Int}}(undef, n_customers[t])
        for k in 1:n_customers[t]
            big_M[t_m][k] = zeros(Int, N + 1)
            sorted_indices = sortperm(utilities[t][k])  # ascending order
            no_buy_index = findfirst(==(N + 1), sorted_indices)
            for (index, i_1) in enumerate(sorted_indices[1:(end - 1)])
                if index >= no_buy_index
                    higher_items = [
                        i_2 for i_2 in sorted_indices[(index + 1):end] if i_2 <= N
                    ]
                    ini_stock_sum = sum(s0[i_2] for i_2 in higher_items)
                    quota_sum = sum(max_q[τ, i_2] for τ in 1:t for i_2 in higher_items)
                    # M = ∑_τ=1^t ∑_{i_2: u_{i_2} > u_{i_1}} max_q[τ][i_2] + stock_ini[i_2] + 1
                    big_M[t_m][k][i_1] = quota_sum + ini_stock_sum + 1
                end
            end
        end
    end
    return big_M
end

function stock_constraints!(m, y, s, α, T, N, nb_customers, stock_ini)
    # Initial stock
    @constraint(m, [i in 1:N], s[1, i] == stock_ini[i])
    # Stock dynamics
    @constraint(
        m,
        [i in 1:N, t in 1:T],
        s[t + 1, i] == s[t, i] + y[t, i] - sum(α[i, t, k] for k in 1:nb_customers[t])
    )
    return nothing
end

function customer_constraints!(m, α, T, N, nb_customers)
    # Each customer buys at most one vehicle (no purchase option included)
    @constraint(
        m, [t in 1:T, k in 1:nb_customers[t]], sum(α[i, t, k] for i in 1:(N + 1)) == 1
    )
    return nothing
end

function sales_order_constraints!(m, y, s, α, T, N, nb_customers, utilities, big_M)
    for t in 1:T
        for k in 1:nb_customers[t]
            sorted_indices = sortperm(utilities[t][k])  # ascending order
            no_buy_index = findfirst(==(N+1), sorted_indices)
            for (index, i_1) in enumerate(sorted_indices[1:(end - 1)])
                # no-buy case
                if index < no_buy_index
                    @constraint(m, α[i_1, t, k] == 0)
                    continue
                else
                    # don't sell i_1 if ∃ i_2 in stock s.t. u_{i_2} > u_{i_1} 
                    if k == 1
                        @constraint(
                            m,
                            α[i_1, t, k] <= (
                                1 -
                                sum(
                                    s[t, i_2] + y[t, i_2] for
                                    i_2 in sorted_indices[(index + 1):end] if i_2 <= N
                                ) / big_M[t][k][i_1]
                            ),
                        )
                    else
                        @constraint(
                            m,
                            α[i_1, t, k] <= (
                                1 -
                                sum(
                                    s[t, i_2] + y[t, i_2] -
                                    sum(α[i_2, t, j] for j in 1:(k - 1)) for
                                    i_2 in sorted_indices[(index + 1):end] if i_2 <= N
                                ) / big_M[t][k][i_1]
                            ),
                        )
                    end
                end
            end
        end
    end
    return nothing
end

"""
$TYPEDSIGNATURES

Add quota constraints to the model.
"""
function quota_constraints!(m, y, T, N, constraints_matrix, quotas)
    nb_cons = size(constraints_matrix, 1)
    @constraint(
        m,
        [c in 1:(nb_cons), t in 1:T],
        sum(constraints_matrix[c, i] * y[t, i] for i in 1:N) <= quotas[t, c]
    )
    return nothing
end

"""
$TYPEDSIGNATURES

Add physical stock constraints (linearization of (x)₊).
"""
function physical_stock_constraints!(
    m, y, α, v, T, N, delivery_delay, stock_ini, nb_customers
)
    @constraint(
        m,
        [i in 1:N, t in (delivery_delay + 1):(T + 1)],
        v[t, i] >=
            stock_ini[i] + sum(y[τ, i] for τ in 1:(t - delivery_delay)) -
        sum(α[i, τ, k] for τ in 1:(t - 1) for k in 1:nb_customers[τ])
    )
    return nothing
end

"""
$TYPEDSIGNATURES

Add stock bounds constraints.
"""
function stock_bounds_constraints!(m, s, T, N, s_min, s_sup, stock_inf, stock_sup)
    # stock Inf
    @constraint(m, [t in 1:T], s_min[t] >= stock_inf - sum(s[t + 1, i] for i in 1:N))
    # stock Sup
    @constraint(m, [t in 1:T], s_sup[t] >= sum(s[t + 1, i] for i in 1:N) - stock_sup)
    return nothing
end

"""
$TYPEDSIGNATURES

Compute the base objective function.
"""
function compute_objective(y, s, α, v, T, s_min, s_sup, env, nb_customers)
    N = item_count(env)
    # margin
    margin = sum(
        prices(env)[i] * sum(α[i, t, k] for t in 1:T for k in 1:nb_customers[t]) for
        i in 1:N
    )
    # virtual stock cost
    virtual_stock = sum(virtual_stock_cost(env)[i] * s[t + 1, i] for t in 1:T for i in 1:N)
    # physical stock cost
    physical_stock = sum(
        physical_stock_cost(env)[i] * v[t, i] for t in 1:(T + 1) for i in 1:N
    )
    # cost under stock min
    under_stock_min = sum(s_min)
    over_stock_sup = sum(s_sup)

    objective =
        margin - virtual_stock - physical_stock -
        over_stock_bound_cost(env) * (under_stock_min + over_stock_sup)

    return objective
end

function solver_variable_to_dataset(
    env::Environment, scenario::Scenario, s_val, y_val, α_val, obj_val; θ=nothing, κ=1.0
)
    s_val = Int.(round.(s_val))      # (T+1, N)
    y_val = Int.(round.(y_val))      # (T, N)
    α_val = Int.(round.(α_val))      # (N+1, T, k)

    config = env.config
    T = max_steps(config) - current_epoch(env) + 1
    N = item_count(config)
    n_customers = nb_customers(scenario)[current_epoch(env):end]
    # sales_full[t, i] = total units of item i sold at epoch t
    sales_full = zeros(Int, T, N)
    for t in 1:T, i in 1:N
        sales_full[t, i] = sum(round(Int, value(α_val[i, t, k])) for k in 1:n_customers[t])
    end
    dataset = Vector{DataSample}(undef, T)

    # initial state, before any replenishment/sales (epoch 0 / pre-action)
    init_state = DRPState(config, s_val[1, :])
    x_init = compute_features(init_state)
    y_init = y_val[1, :]
    init_state.current_cost = compute_cost(init_state, y_init, sales_full[1, :])
    dataset[1] = DataSample(;
        y=y_init,
        x=x_init,
        state=init_state,
        next_sales=sales_full[1, :],
        customers=n_customers[1],
    )
    for t in 2:T
        state_t = DRPState(;
            config=config,
            current_epoch=t,
            stock=s_val[t, :],
            stock_history=s_val[1:t, :],
            replenishment_history=y_val[1:(t - 1), :],
            sales_history=sales_full[1:(t - 1), :],
            customer_history=n_customers[1:(t - 1)],
            ub_per_item=s_val[t, :] .+ max_quotas(config)[t, :],
            current_cost=0.0,
        )
        y_true = y_val[t, :]
        state_t.current_cost = compute_cost(state_t, y_true, sales_full[t, :])
        x = compute_features(state_t)
        dataset[t] = DataSample(;
            y=y_true,
            x,
            state=state_t,
            next_sales=sales_full[t, :],
            customers=n_customers[t],
        )
    end

    final_obj_val = dataset[end].state.current_cost
    if !isnothing(θ)
        if typeof(θ) == Vector{Float32}
            final_obj_val += κ * dot(θ, g(dataset[1].y; state=dataset[1].state))
        end
    end
    @assert obj_val ≈ final_obj_val
    return dataset
end

"""
$TYPEDSIGNATURES

Construct yη vector for 
"""
function g_model(m, N, ub, y, s)
    @variable(m, y_eta[i in 1:N, k in 1:ub[i]] >= 0, Int)

    @constraint(m, [i in 1:N], y_eta[i, 1] <= 1)
    @constraint(m, [i in 1:N], y_eta[i, 1] * ub[i] >= s[i] + y[i])
    @constraint(m, [i in 1:N], y_eta[i, 1] <= s[i] + y[i])

    @constraint(m, [i in 1:N, k in 2:ub[i]], y_eta[i, k] >= s[i] + y[i] - (k - 1))

    y_eta_vec = Vector{AffExpr}(undef, sum(ub))
    row = 1
    for i in 1:N
        y_eta_vec[row] = 1 * y_eta[i, 1]
        for k in 2:ub[i]
            y_eta_vec[row + k - 1] = -y_eta[i, k]
        end
        row += ub[i]
    end
    return vcat(vec(y), y_eta_vec)
end

"""
$TYPEDSIGNATURES

Solve the anticipative problem for a given instance and scenario.
"""
function anticipative_solver(
    env::Environment,
    rng::AbstractRNG,
    scenario::Scenario=env.scenario;
    model_builder=highs_model,
    reset_env::Bool=true,
    verbose::Bool=false,
    big_M=nothing,
    θ=nothing,
    state::DRPState=env.state,
    κ::Float64=1.0,
    mip_gap::Float64=0.0,
)
    if reset_env
        reset!(env, rng)
        scenario = env.scenario
        state = env.state
    end

    if big_M === nothing
        big_M = compute_bigM!(env, scenario)
    end

    @assert !is_terminated(env)

    m = model_builder()
    verbose || set_silent(m)
    set_attribute(m, MOI.RelativeGapTolerance(), mip_gap)
    N = item_count(env)
    T = max_steps(env) - current_epoch(env) + 1
    n_customers = nb_customers(scenario)[current_epoch(env):end]
    s0 = stock(env)
    ## Variables
    @variable(m, y[1:T, 1:N] >= 0, Int) # replenishments
    @variable(m, s[1:(T + 1), 1:N] >= 0, Int) # stock
    @variable(m, α[i in 1:(N + 1), t in 1:T, k in 1:n_customers[t]], Bin) # sales
    @variable(m, v[1:(T + 1), 1:N] >= 0, Int) # physical stock
    @variable(m, s_min[1:T] >= 0, Int) # stock under min
    @variable(m, s_sup[1:T] >= 0, Int) # stock over max

    ## Constraints
    stock_constraints!(m, y, s, α, T, N, n_customers, s0)
    customer_constraints!(m, α, T, N, n_customers)
    sales_order_constraints!(
        m, y, s, α, T, N, n_customers, scenario.utilities[current_epoch(env):end], big_M
    )
    quota_constraints!(m, y, T, N, constraints_matrix(env), quotas(env))
    physical_stock_constraints!(m, y, α, v, T, N, delivery_delay(env), s0, n_customers)
    stock_bounds_constraints!(m, s, T, N, s_min, s_sup, stock_inf(env), stock_sup(env))

    ## Objective
    objective = compute_objective(y, s, α, v, T, s_min, s_sup, env, n_customers)
    if θ !== nothing
        g_y = g_model(m, N, ub_per_item(state), y[1, :], s[1, :])
        @assert length(θ) == N + sum(ub_per_item(state))
        objective += κ * dot(θ, g_y)
    end
    @objective(m, Max, objective)

    optimize!(m)
    if primal_status(m) == MOI.FEASIBLE_POINT
        obj_val = objective_value(m)
        dataset = solver_variable_to_dataset(
            env, scenario, value.(s), value.(y), value.(α), obj_val; θ=θ, κ=κ
        )
        return obj_val, dataset
    else
        @warn("No feasible points found.")
        return nothing, nothing
    end
end
