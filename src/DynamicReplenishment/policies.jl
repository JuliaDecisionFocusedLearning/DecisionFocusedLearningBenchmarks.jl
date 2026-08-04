# policy utils

function mean_feature_order(env::Environment; rng::AbstractRNG=Xoshiro(nothing))
    item_features = features(env)   # (d, N), price is a separate field
    # if no features other than price, return a random order
    if size(item_features, 1) == 0
        return randperm(rng, item_count(env))
    end
    # else order items by the mean of their features
    return sortperm(vec(mean(item_features; dims=1)))
end

function max_quotas_item_after_repl(
    item_idx::Int,
    replenishment::Vector{Int},
    time_idx::Int,
    quotas::Matrix{Int},
    cons_mat::Matrix{Int},
)
    N = size(cons_mat, 2)
    nb_constraints = size(cons_mat, 1)
    return max(
        0,
        minimum([
            quotas[time_idx, c] - sum(replenishment[j] * cons_mat[c, j] for j in 1:N) for
            c in 1:nb_constraints if cons_mat[c, item_idx] == 1
        ]),
    )
end

# Policies 

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

function random_policy(env::Environment, rng::AbstractRNG=Xoshiro(nothing))
    N = item_count(env)
    cons_mat = constraints_matrix(env)
    q = quotas(env)
    replenishment = zeros(Int, N)
    order_item = randperm(rng, N)
    t = current_epoch(env)
    for item in order_item
        max_quota_item = max_quotas_item_after_repl(item, replenishment, t, q, cons_mat)
        replenishment[item] = rand(rng, 0:max_quota_item)
    end
    return replenishment
end

function mean_anticipative_policy(
    env::Environment;
    rng::AbstractRNG=Xoshiro(nothing),
    anticipative_results::AbstractVector{<:DataSample}=DataSample[],
    order_item::Function=mean_feature_order,
)
    if isempty(anticipative_results)
        @warn "mean_anticipative_policy: no anticipative results provided, falling back to lazy policy"
        return lazy_policy(env)
    end
    N = item_count(env)
    mean_anticipative_replenishment = zeros(Float64, N)
    for sample in anticipative_results
        mean_anticipative_replenishment .+= sample.y
    end
    mean_anticipative_replenishment ./= length(anticipative_results)
    replenishment = zeros(Int, N)
    t = current_epoch(env)
    q = quotas(env)
    cons_mat = constraints_matrix(env)
    for item in order_item(env; rng=rng)
        max_quota_item = max_quotas_item_after_repl(item, replenishment, t, q, cons_mat)
        replenishment[item] = min(
            round(Int, mean_anticipative_replenishment[item]), max_quota_item
        )
    end
    return replenishment
end

"""
$TYPEDEF

Callable wrapping [`mean_anticipative_policy`](@ref) with a dataset of anticipative results.
"""
struct MeanAnticipativePolicyCall
    anticipative_results::Vector{DataSample}
    order_item::Function
end

function (p::MeanAnticipativePolicyCall)(env::Environment; kwargs...)
    return mean_anticipative_policy(
        env; kwargs..., anticipative_results=p.anticipative_results, order_item=p.order_item
    )
end

"""
$TYPEDSIGNATURES

Solve a sample average approximation of the replenishment problem over the remaining horizon
on `nb_scenarios` sampled scenarios, and return the first-stage replenishment decision.

The first replenishment is constrained to be identical across scenarios, so the returned
decision is implementable without knowing the realized demand. When `θ` is given, the
objective is augmented with `κ * dot(θ, g(y))` to bias the decision towards the predicted
utilities.

The solver is stopped after `time_limit` seconds (10 minutes by default), returning the
best feasible solution found so far; pass `time_limit=nothing` to disable it. If no feasible
point was found at all, the policy falls back to replenishing nothing.

`warm_start` hands the solver an initial all-zero replenishment.
"""
function saa_policy(
    env::Environment;
    nb_scenarios::Int=5,
    rng::AbstractRNG=Xoshiro(nothing),
    model_builder=highs_model,
    verbose::Bool=false,
    mip_gap::Float64=0.0,
    time_limit::Union{Real,Nothing}=600.0,
    warm_start::Bool=true,
    θ=nothing,
    state::DRPState=env.state,
    κ::Float64=1.0,
)
    scenarios = [generate_scenario(env.config; rng=rng) for _ in 1:nb_scenarios]
    bigM_s = [compute_bigM_sales(env, scenario) for scenario in scenarios]
    bigM_ps = [compute_bigM_physical_stock(env, scenario) for scenario in scenarios]

    @assert !is_terminated(env)

    m = model_builder()
    verbose || set_silent(m)
    set_attribute(m, MOI.RelativeGapTolerance(), mip_gap)
    isnothing(time_limit) || set_attribute(m, MOI.TimeLimitSec(), Float64(time_limit))
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
            bigM_ps[s_idx],
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

    # Warm start: replenishing nothing is always feasible
    if warm_start
        for s_idx in 1:nb_scenarios, t in 1:T, i in 1:N
            set_start_value(y[s_idx, t, i], 0)
        end
    end

    optimize!(m)
    if primal_status(m) == MOI.FEASIBLE_POINT
        return round.(Int, value.(y[1, 1, :]))
    else
        # Fall back to a lazy decision instead of `nothing`
        @warn "SAA: no feasible point found, falling back to lazy replenishment"
        return zeros(Int, N)
    end
end

"""
$TYPEDEF

Callable wrapping [`saa_policy`](@ref) with a fixed `model_builder`.
"""
struct SAAPolicyCall{M}
    model_builder::M
    verbose::Bool
    mip_gap::Float64
    nb_scenarios::Int
    "solver time limit in seconds, `nothing` to disable"
    time_limit::Union{Float64,Nothing}
    "hand the solver an all-zero replenishment as initial solution"
    warm_start::Bool
end

function SAAPolicyCall(
    model_builder::M;
    verbose::Bool=false,
    mip_gap::Float64=1e-2,
    nb_scenarios::Int=1,
    time_limit::Union{Real,Nothing}=600.0,
    warm_start::Bool=true,
) where {M}
    return SAAPolicyCall{M}(
        model_builder,
        verbose,
        mip_gap,
        nb_scenarios,
        isnothing(time_limit) ? nothing : Float64(time_limit),
        warm_start,
    )
end

function (p::SAAPolicyCall)(env::Environment; kwargs...)
    return saa_policy(
        env;
        kwargs...,
        model_builder=p.model_builder,
        verbose=p.verbose,
        mip_gap=p.mip_gap,
        nb_scenarios=p.nb_scenarios,
        time_limit=p.time_limit,
        warm_start=p.warm_start,
    )
end
