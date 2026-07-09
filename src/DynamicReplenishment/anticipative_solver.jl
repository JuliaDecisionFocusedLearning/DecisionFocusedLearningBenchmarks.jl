
"""
$TYPEDSIGNATURES

Compute big M values for a scenario of a specific environment.
"""
function compute_bigM!(env::Environment, scenario::Scenario)
    T = max_steps(env.config)
    N = item_count(env.config)
    max_q = max_quotas(env.config)
    s0 = stock_ini(env)
    nb_customers = scenario.nb_customers
    utilities = scenario.utilities
    big_M = Vector{Vector{Vector{Int}}}(undef, T)
    for t in 1:T
        big_M[t] = Vector{Vector{Int}}(undef, nb_customers[t])
        for k in 1:nb_customers[t]
            big_M[t][k] = zeros(Int, N + 1)
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
                    big_M[t][k][i_1] = quota_sum + ini_stock_sum + 1
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
function compute_objective(y, s, α, v, s_min, s_sup, env, nb_customers)
    N = item_count(env)
    T = max_steps(env)
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
    env::Environment, scenario::Scenario, s_val, y_val, α_val, obj_val
)
    s_val = Int.(round.(s_val))      # (T+1, N)
    y_val = Int.(round.(y_val))      # (T, N)
    α_val = Int.(round.(α_val))      # (N+1, T, k)

    config = env.config
    T = max_steps(config)
    N = item_count(config)
    # sales_full[t, i] = total units of item i sold at epoch t
    sales_full = zeros(Int, T, N)
    for t in 1:T, i in 1:N
        sales_full[t, i] = sum(
            round(Int, value(α_val[i, t, k])) for k in 1:scenario.nb_customers[t]
        )
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
        customers=scenario.nb_customers[1],
    )
    for t in 2:T
        state_t = DRPState(;
            config=config,
            current_epoch=t,
            stock=s_val[t, :],
            stock_history=s_val[1:t, :],
            replenishment_history=y_val[1:(t - 1), :],
            sales_history=sales_full[1:(t - 1), :],
            customer_history=scenario.nb_customers[1:(t - 1)],
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
            customers=scenario.nb_customers[t],
        )
    end

    final_state = dataset[end].state
    @assert obj_val ≈ final_state.current_cost

    return dataset
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
    reset_env=true,
    verbose=false,
    big_M=nothing,
)
    if reset_env
        reset!(env, rng)
        scenario = env.scenario
    end

    if big_M === nothing
        big_M = compute_bigM!(env, scenario)
    end

    @assert !is_terminated(env)

    m = model_builder()
    verbose || set_silent(m)
    N = item_count(env)
    T = max_steps(env)
    nb_customers = scenario.nb_customers
    s0 = stock_ini(env)
    ## Variables
    @variable(m, y[1:T, 1:N] >= 0, Int) # replenishments
    @variable(m, s[1:(T + 1), 1:N] >= 0, Int) # stock
    @variable(m, α[i in 1:(N + 1), t in 1:T, k in 1:nb_customers[t]], Bin) # sales
    @variable(m, v[1:(T + 1), 1:N] >= 0, Int) # physical stock
    @variable(m, s_min[1:T] >= 0, Int) # stock under min
    @variable(m, s_sup[1:T] >= 0, Int) # stock over max

    ## Constraints
    stock_constraints!(m, y, s, α, T, N, nb_customers, s0)
    customer_constraints!(m, α, T, N, nb_customers)
    sales_order_constraints!(m, y, s, α, T, N, nb_customers, scenario.utilities, big_M)
    quota_constraints!(m, y, T, N, constraints_matrix(env), quotas(env))
    physical_stock_constraints!(m, y, α, v, T, N, delivery_delay(env), s0, nb_customers)
    stock_bounds_constraints!(m, s, T, N, s_min, s_sup, stock_inf(env), stock_sup(env))

    ## Objective
    objective = compute_objective(y, s, α, v, s_min, s_sup, env, nb_customers)
    @objective(m, Max, objective)

    optimize!(m)
    if primal_status(m) == MOI.FEASIBLE_POINT
        if termination_status(m) != MOI.OPTIMAL
            @warn("Optimal not found")
        end
        obj_val = JuMP.objective_value(m)
        ## generate datasample from solution ==> compute features ...
        state = solver_variable_to_dataset(
            env, scenario, value.(s), value.(y), value.(α), obj_val
        )
        return JuMP.objective_value(m), state
    else
        write_to_file(m, "single_scenario_oracle.lp")
        println("Not optimal")
        return nothing, nothing
    end
end
