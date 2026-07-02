"""
$TYPEDSIGNATURES

Solve the Replenishment Problem defined by the config and cost vectors θ and η.
"""
function replenishment_problem(
    Θ; state::DRPState, y_true=nothing, model_builder=highs_model
)
    config = state.config
    N = item_count(config)
    ub_same_item = UB_item(config)
    t = current_epoch(state)

    θ = Θ[1:N]
    η = reshape(Θ[(1 + N):end], N, ub_same_item)
    m = model_builder()
    set_silent(m)
    # Variables
    # number of archetypes replenished
    ### TODO: check the definition of the variables: we could use max_quotas instead of ub_same_item
    @variable(m, y[1:N, 1:ub_same_item], Bin)
    # penalization
    @variable(m, z[1:N, 1:ub_same_item], Bin)

    # Objective function
    ## TODO : review the definition of the objective function 
    ## ==> 1. the over stock cost is paid only after the sales 
    ## ==> 2. Here we do not have a piecewise concave function exactly like we would like I think (maybe don't have a theta ? )
    utility_reward = sum(θ[i] * sum(y[i, :]) for i in 1:N)
    stock_penalization = sum(
        η[i, 1] * sum(z[i, :]) -
        sum(z[i, j] * sum(η[i, k] for k in 2:j) for j in 2:ub_same_item) for i in 1:N
    )
    @objective(m, Max, utility_reward + stock_penalization)
    # Constraints
    ## penalization constraints
    @constraint(
        m,
        [i in 1:N],
        sum(y[i, j] for j in 1:ub_same_item) + state.stock[i] ==
            sum(z[i, j] for j in 1:ub_same_item)
    )
    ## quota constraints
    @constraint(
        m,
        [c in 1:nb_constraints(config)],
        sum(config.constraints_matrix[c, i] * y[i, j] for i in 1:N, j in 1:ub_same_item) <=
            config.quotas[t, c]
    )
    ## structural constraints
    @constraint(m, [i in 1:N, j in 1:(ub_same_item - 1)], y[i, j] >= y[i, j + 1])
    @constraint(m, [i in 1:N, j in 1:(ub_same_item - 1)], z[i, j] >= z[i, j + 1])

    if y_true !== nothing
        y_candidate = y_true[:, 1:ub_same_item]
        z_candidate = y_true[:, (1 + ub_same_item):end]
        for i in 1:N
            for j in 1:ub_same_item
                fix(y[i, j], y_candidate[i, j]; force=true)
                fix(z[i, j], z_candidate[i, j]; force=true)
            end
        end
    end

    optimize!(m)

    if primal_status(m) == MOI.FEASIBLE_POINT
        if termination_status(m) != MOI.OPTIMAL
            @warn("Optimal not found")
        end
        final_vec = hcat(value.(y), value.(z))
        return final_vec
    else
        write_to_file(m, "replenishment_problem_infeasible.lp")
        error("The model did not find an optimal or feasible solution.")

        return nothing, nothing
    end
end

"""
$TYPEDSIGNATURES

Transform a replenishment and stock into a y solution for the replenishment problem.
"""
function y_oracle(env::Environment, replenishment, stock; verbose=false)
    N = item_count(env)
    ub_item = UB_item(env)
    y = zeros(Float64, N, ub_item)
    z = zeros(Float64, N, ub_item)
    for i in 1:N
        try
            y[i, 1:replenishment[i]] .= 1.0
        catch
            @error(
                "Error in y_oracle: replenishment[i]=$(replenishment[i]), stock[i]=$(stock[i]), ub_item=$ub_item"
            )
        end
        z[i, 1:(replenishment[i] + stock[i])] .= 1.0
    end
    return hcat(y, z)
end

function g(y; state::DRPState, kwargs...)
    config = state.config
    N = item_count(config)
    ub_same_item = UB_item(config)
    yθ = [sum(y[i, 1:ub_same_item]) for i in 1:N]  # shape (1, N)
    z = y[:, (ub_same_item + 1):end] # shape (N, ub_same_item)
    # Build yη with column-major ordering to match reshape of η (N × ub)
    # This ensures <Θ, g(y)> aligns with the MILP objective using η[i,k].
    yη = [(
        if k == 1
            sum(z[i, j] for j in k:ub_same_item)
        else
            -sum(z[i, j] for j in k:ub_same_item)
        end
    ) for i in 1:N, k in 1:ub_same_item] # Matrix (n, ub)

    return vcat(vec(yθ), vec(yη))
end

function get_replenishment_from_y(y; state::DRPState)
    config = state.config
    N = item_count(config)
    ub_same_item = UB_item(config)
    replenishment = round.(Int, [sum(y[i, 1:ub_same_item]) for i in 1:N])
    return replenishment
end