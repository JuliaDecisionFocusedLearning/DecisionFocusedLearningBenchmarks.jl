function greedy_policy(env::Environment; model_builder=highs_model)
    _, state = observe(env)
    N = item_count(env)
    ub = ub_per_item(env)
    Θ = zeros(N + sum(ub))
    Θ[1:N] .= prices(env)
    return (replenishment_problem(Θ; state, model_builder=model_builder))
end

function lazy_policy(env::Environment)
    N = item_count(env)
    return zeros(Int, N)
end

function random_policy(env::Environment, rng::AbstractRNG=Xoshiro(0))
    N = item_count(env)
    cons_mat = constraints_matrix(env)
    q = quotas(env)
    replenishment = zeros(Int, N)
    order_item = randperm(rng, N)
    t = current_epoch(env)
    for item in order_item
        max_quota_item = max(
            0,
            minimum([
                q[t, c] - sum(replenishment[j] * cons_mat[c, j] for j in 1:N) for
                c in 1:nb_constraints(env.config) if cons_mat[c, item] == 1
            ]),
        )
        replenishment[item] = rand(0:max_quota_item)
    end
    return replenishment
end

function saa_policy(
    env::Environment;
    nb_scenarios::Int=5,
    rng::AbstractRNG=Xoshiro(0),
    model_builder=highs_model,
    verbose::Bool=false,
    mip_gap::Float64=0.0,
    θ=nothing,
    state::DRPState=env.state,
    κ::Float64=1.0,
)
    scenarios = [generate_scenario(env.config; rng=rng) for _ in 1:nb_scenarios]
    bigM_s = [compute_bigM_sales!(env, scenario) for scenario in scenarios]
    bigM_ps = compute_bigM_physical_stock!(env)

    @assert !is_terminated(env)

    m = model_builder()
    verbose || set_silent(m)
    set_attribute(m, MOI.RelativeGapTolerance(), mip_gap)
    N = item_count(env)
    T = max_steps(env) - current_epoch(env) + 1
    n_customers = [nb_customers(scenario)[current_epoch(env):end] for scenario in scenarios]
    s0 = stock(env)
    ## Variables
    @variable(m, y[1:nb_scenarios, 1:T, 1:N] >= 0, Int) # replenishments
    @variable(m, s[1:nb_scenarios, 1:(T + 1), 1:N] >= 0, Int) # stock
    @variable(
        m,
        α[s_idx in 1:nb_scenarios, i in 1:(N + 1), t in 1:T, k in 1:n_customers[s_idx][t]],
        Bin
    ) # sales
    @variable(m, v[1:nb_scenarios, 1:(T + 1), 1:N] >= 0, Int) # physical stock
    @variable(m, z[1:nb_scenarios, 1:N, 2:(T + 1)], Bin) # auxiliary binary for physical stock linearization
    @variable(m, s_min[1:nb_scenarios, 1:T] >= 0, Int) # stock under min
    @variable(m, s_sup[1:nb_scenarios, 1:T] >= 0, Int) # stock over max

    ## Constraints
    @constraint(m, [s_idx in 2:nb_scenarios, i in 1:N], y[1, 1, i] == y[s_idx, 1, i]) # first replenishment is the same for all scenarios
    objective = zero(AffExpr)
    for s_idx in 1:nb_scenarios
        stock_constraints!(
            m,
            y[s_idx, :, :],
            s[s_idx, :, :],
            α[s_idx, :, :, :],
            T,
            N,
            n_customers[s_idx],
            s0,
        )
        customer_constraints!(m, α[s_idx, :, :, :], T, N, n_customers[s_idx])
        sales_order_constraints!(
            m,
            y[s_idx, :, :],
            s[s_idx, :, :],
            α[s_idx, :, :, :],
            T,
            N,
            n_customers[s_idx],
            scenarios[s_idx].utilities[current_epoch(env):end],
            bigM_s[s_idx],
        )
        quota_constraints!(m, y[s_idx, :, :], T, N, constraints_matrix(env), quotas(env))
        physical_stock_constraints!(
            m,
            y[s_idx, :, :],
            α[s_idx, :, :, :],
            v[s_idx, :, :],
            z[s_idx, :, :],
            T,
            N,
            delivery_delay(env),
            s0,
            n_customers[s_idx],
            bigM_ps,
        )
        stock_bounds_constraints!(
            m,
            v[s_idx, :, :],
            T,
            N,
            s_min[s_idx, :],
            s_sup[s_idx, :],
            stock_inf(env),
            stock_sup(env),
        )
        ## Objective
        objective += compute_objective(
            y[s_idx, :, :],
            s[s_idx, :, :],
            α[s_idx, :, :, :],
            v[s_idx, :, :],
            T,
            s_min[s_idx, :],
            s_sup[s_idx, :],
            env,
            n_customers[s_idx],
        )
    end

    if θ !== nothing
        g_y = g_model(m, N, ub_per_item(state), y[1, 1, :], s[1, 1, :])
        @assert length(θ) == N + sum(ub_per_item(state))
        objective += κ * dot(θ, g_y)
    end
    @objective(m, Max, objective)

    optimize!(m)
    if primal_status(m) == MOI.FEASIBLE_POINT
        return round.(Int, value.(y[1, 1, :]))
    else
        @warn("No feasible points found.")
        return nothing
    end
end

