function _obj_function(N::Int, ub::Vector{Int}, Θ, y, z)
    θ = Θ[1:N]
    η = Vector{Vector{Float64}}(undef, N)
    offset = N
    for i in 1:N
        η[i] = Θ[(offset + 1):(offset + ub[i])]
        offset += ub[i]
    end
    utility_reward = sum(θ[i] * y[i] for i in 1:N)
    stock_penalization = sum(
        η[i][1] * z[i, 1] - sum(z[i, j] * sum(η[i][k] for k in 2:j) for j in 2:ub[i]) for
        i in 1:N
    )
    return utility_reward + stock_penalization
end

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

    m = model_builder()
    set_silent(m)
    # Variables
    @variable(m, y[1:N], Bin)
    # penalization
    @variable(m, z[i in 1:N, j in 1:ub[i]], Bin)

    # Objective function
    ## TODO : review the definition of the objective function 
    ## We do not have a piecewise concave function exactly like we would like I think (maybe don't have a theta ? )
    @objective(m, Max, _obj_function(N, ub, Θ, y, z))
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
        z_true = get_z_from_y(y_true, state)
        for i in 1:N
            fix(y[i], y_true[i]; force=true)
            for j in 1:ub[i]
                fix(z[i, j], z_true[i, j]; force=true)
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
    yη = zeros(sum(ub))
    row = 1
    for i in 1:N
        for k in 1:ub[i]
            if k == 1
                yη[row] = stock_and_replenishment[i] > 0 ? 1 : 0
            else
                yη[row + k - 1] = -max(0, stock_and_replenishment[i] - (k - 1))
            end
        end
        row += ub[i]
    end
    return vcat(vec(yθ), vec(yη))
end

function get_z_from_y(y_true::Vector{Int}, state::DRPState)
    N = length(y_true)
    ub = ub_per_item(state)
    z_true = zeros(Int, N, maximum(ub))
    for i in 1:N
        stock_and_replenishment = round(Int(state.stock[i] + y_true[i]))
        z_true[i, 1:stock_and_replenishment] .= 1
    end
    return z_true
end
