"""
$TYPEDSIGNATURES

Solve the Replenishment Problem defined by the config and cost vectors θ and η.
"""
function replenishment_problem(
    Θ; state::DRPState, y_true=nothing, model_builder=highs_model
)
    config = state.config
    N = item_count(config)
    ub = ub_per_item(state)
    t = current_epoch(state)

    θ = Θ[1:N]
    η = Vector{Vector{Float64}}(undef, N)
    offset = N
    for i in 1:N
        η[i] = Θ[(offset + 1):(offset + ub[i])]
        offset += ub[i]
    end
    m = model_builder()
    set_silent(m)
    # Variables
    @variable(m, y[1:N], Bin)
    # penalization
    @variable(m, z[i in 1:N, j in 1:ub[i]], Bin)

    # Objective function
    ## TODO : review the definition of the objective function 
    ## We do not have a piecewise concave function exactly like we would like I think (maybe don't have a theta ? )
    utility_reward = sum(θ[i] * y[i] for i in 1:N)
    stock_penalization = sum(
        η[i][1] * sum(z[i, j] for j in 1:ub[i]) -
        sum(z[i, j] * sum(η[i][k] for k in 2:j) for j in 2:ub[i]) for i in 1:N
    )
    @objective(m, Max, utility_reward + stock_penalization)
    # Constraints
    ## penalization constraints
    @constraint(m, [i in 1:N], y[i] + state.stock[i] == sum(z[i, j] for j in 1:ub[i]))
    ## quota constraints
    @constraint(
        m,
        [c in 1:nb_constraints(config)],
        sum(config.constraints_matrix[c, i] * y[i] for i in 1:N) <= config.quotas[t, c]
    )
    ## structural constraints
    @constraint(m, [i in 1:N, j in 1:(ub[i] - 1)], z[i, j] >= z[i, j + 1])

    if y_true !== nothing
        z_candidate = get_z_from_y(y_true; state)
        for i in 1:N
            fix(y[i], y_candidate[i]; force=true)
            for j in 1:ub[i]
                fix(z[i, j], z_candidate[i, j]; force=true)
            end
        end
    end

    optimize!(m)

    if primal_status(m) == MOI.FEASIBLE_POINT
        if termination_status(m) != MOI.OPTIMAL
            @warn("Optimal not found")
        end
        return Int.(round.(value.(y)))
    else
        write_to_file(m, "replenishment_problem_infeasible.lp")
        error("The model did not find an optimal or feasible solution.")

        return nothing, nothing
    end
end

function g(y; state::DRPState, kwargs...)
    config = state.config
    N = item_count(config)
    ub = ub_per_item(state)
    yθ = copy(y)  # shape (1, N)
    stock_and_replenishment = state.stock .+ y
    yη = [(
        if k == 1
            max(0, stock_and_replenishment[i] - (k-1))
        else
            -max(0, stock_and_replenishment[i] - (k-1))
        end
    ) for i in 1:N, k in 1:ub[i]] # shape (N, ub[i])
    return vcat(vec(yθ), vec(yη))
end
