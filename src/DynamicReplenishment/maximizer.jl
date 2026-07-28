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
    @variable(m, 0 <= y[i in 1:N] <= ub[i], Int)
    # penalization
    @variable(m, z[i in 1:N, j in 1:ub[i]], Bin)

    # Objective function
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

    if !isnothing(y_true)
        z_true = get_z_from_y(y_true, state)
        for i in 1:N
            fix(y[i], y_true[i]; force=true)
            for j in 1:ub[i]
                fix(z[i, j], z_true[i, j]; force=true)
            end
        end
    end

    optimize!(m)

    return Int.(round.(value.(y)))
end

function g(y; state::DRPState, kwargs...)
    N = item_count(state.config)
    ub = ub_per_item(state)
    stock_and_replenishment = round.(Int, state.stock .+ y)
    yη = Vector{Float64}(undef, sum(ub))
    row = 1
    for i in 1:N
        yη[row] = stock_and_replenishment[i] > 0 ? 1 : 0
        for k in 2:ub[i]
            yη[row + k - 1] = -max(0, stock_and_replenishment[i] - (k - 1))
        end
        row += ub[i]
    end
    return vcat(vec(y), vec(yη))
end

function get_z_from_y(y_true::Vector{Int}, state::DRPState)
    N = length(y_true)
    ub = ub_per_item(state)
    stock_and_replenishment = round.(Int, state.stock .+ y_true)
    z_true = zeros(Int, N, maximum(ub))
    for i in 1:N
        z_true[i, 1:min(ub[i], stock_and_replenishment[i])] .= 1
    end
    return z_true
end
