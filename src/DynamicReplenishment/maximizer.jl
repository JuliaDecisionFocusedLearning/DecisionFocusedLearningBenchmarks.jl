"""
$TYPEDSIGNATURES

Stock penalization `Σ_i Σ_j z[i,j] Σ_{k≤j} η[i][k]`, read on the COMPACT `η` block.
"""
_eta_penalty(::NoEta, ηc, ub, z, N) = zero(eltype(ηc))

# `Σ_j z[i,j]·(j·η_i) = η_i Σ_j j z[i,j]` : une seule pente par item.
_eta_penalty(::SlopeEta, ηc, ub, z, N) =
    sum(ηc[i] * sum(j * z[i, j] for j in 1:ub[i]) for i in 1:N)

function _eta_penalty(::PiecewiseConstantEta, ηc, ub, z, N)
    offsets = [0; cumsum(ub)[1:(end-1)]]
    return sum(
        sum(z[i, j] * sum(ηc[offsets[i]+k] for k in 1:j) for j in 1:ub[i]) for i in 1:N
    )
end

"""
$TYPEDSIGNATURES

Maximizer objective: margin `θ·y` minus stock penalization.

`Θ = [θ (N) ; η (nb_eta(p, ub))]`. Cumulative sum starts at `k = 1`, so
`η[i][1]` already penalizes the first unit (`m_i(1) = θ_i − η_i[1]`) — this
makes `m_i(j) − m_i(j+1) = η[i][j+1] ≥ 0`, hence automatic concavity.
"""
function _obj_function(N::Int, ub::Vector{Int}, Θ, y, z, p::EtaParametrization)
    θ = Θ[1:N]
    ηc = Θ[(N+1):end]
    return sum(θ[i] * y[i] for i in 1:N) - _eta_penalty(p, ηc, ub, z, N)
end

"""
$TYPEDSIGNATURES

Solve the Replenishment Problem defined by the config and cost vectors θ and η.
"""
function replenishment_problem(
    Θ; state::DRPState, y_true=nothing, model_builder=highs_model,
    parametrization::EtaParametrization=PiecewiseConstantEta(),
)
    config = state.config
    N = item_count(config)
    ub = ub_per_item(state)
    t = current_epoch(state)
    length(Θ) == N + nb_eta(parametrization, ub) || throw(DimensionMismatch(
        "Θ is of length $(length(Θ)) instead of $(N + nb_eta(parametrization, ub)) for " *
            "$(parametrization) : the statistical model and the maximizer do not use the same parametrization of η (or a stored model from before the refactoring is being used)."))

    if !all(isfinite, Θ) || maximum(abs, Θ) > 1e12
        @warn "Maximiser: Θ infinite" max_abs_theta = maximum(abs, Θ) nonfinite_theta = count(
            !isfinite, Θ
        ) maxlog = 10
    end

    m = model_builder()
    set_silent(m)
    # Variables
    @variable(m, 0 <= y[i in 1:N] <= ub[i], Int)
    # penalization
    @variable(m, z[i in 1:N, j in 1:ub[i]], Bin)

    # Objective function
    @objective(m, Max, _obj_function(N, ub, Θ, y, z, parametrization))
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
    @constraint(m, [i in 1:N, j in 1:(ub[i]-1)], z[i, j] >= z[i, j+1])

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

    if primal_status(m) == MOI.FEASIBLE_POINT
        obj = objective_value(m)
        if !isfinite(obj) || abs(obj) >= 1e20
            @warn "Maximiser: infinite objective" objective = obj max_abs_theta = maximum(
                abs, Θ
            ) nonfinite_theta = count(!isfinite, Θ) maxlog = 10
        end
    end

    return Int.(round.(value.(y)))
end

"""
$TYPEDSIGNATURES

Returns the feature vector `g(y)` for the Fenchel-Young map, where `⟨g(y), Θ⟩` is exactly the objective of the maximizer at `y`.

The `η` part has the same compact length as `Θ`, otherwise the scalar product no longer equals the objective and the gradient of the loss is wrong.

With `x_i = s_i + y_i` :

    full    gη[offset_i + k] = −max(0, x_i − k + 1)            (k = 1..ub_i)
    slope   gη[i]            = −Σ_j j·[j ≤ x_i] = −x_i(x_i+1)/2
    none    (empty vector)
"""
function g(y; state::DRPState, parametrization::EtaParametrization=PiecewiseConstantEta(), kwargs...)
    N = item_count(state.config)
    ub = ub_per_item(state)
    x = round.(Int, state.stock .+ y)
    return vcat(vec(y), _eta_features(parametrization, x, ub, N))
end

_eta_features(::NoEta, x, ub, N) = Float64[]

_eta_features(::SlopeEta, x, ub, N) =
    [-0.5 * min(x[i], ub[i]) * (min(x[i], ub[i]) + 1) for i in 1:N]

function _eta_features(::PiecewiseConstantEta, x, ub, N)
    yη = Vector{Float64}(undef, sum(ub))
    row = 1
    for i in 1:N
        for k in 1:ub[i]
            yη[row+k-1] = -max(0, x[i] - (k - 1))
        end
        row += ub[i]
    end
    return yη
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
