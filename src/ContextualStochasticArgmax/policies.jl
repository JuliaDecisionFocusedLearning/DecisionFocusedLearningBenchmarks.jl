"""
$TYPEDSIGNATURES

SAA baseline policy: returns `argmax(mean(scenarios))`.
For a linear argmax problem this is the exact SAA-optimal decision.
Returns a single labeled [`DataSample`](@ref) with `extra=(; scenarios)`.
"""
function csa_saa_policy(ctx_sample, scenarios)
    y = one_hot_argmax(mean(scenarios))
    return [
        DataSample(;
            ctx_sample.context...,
            x=ctx_sample.x,
            y=y,
            extra=(; ctx_sample.extra..., scenarios),
        ),
    ]
end

"""
$TYPEDSIGNATURES

Return the named baseline policies for [`ContextualStochasticArgmaxBenchmark`](@ref).
Each policy has signature `(ctx_sample, scenarios) -> Vector{DataSample}`.
"""
function Utils.generate_baseline_policies(::ContextualStochasticArgmaxBenchmark)
    return (; saa=Policy("SAA", "argmax of mean scenarios", csa_saa_policy))
end

"""
$TYPEDEF

A policy that acts with perfect information about the future scenario.
"""
struct AnticipativeSolver end

function Base.show(io::IO, ::AnticipativeSolver)
    return print(io, "Anticipative solver for ContextualStochasticArgmaxBenchmark")
end

"""
$TYPEDSIGNATURES

Evaluate the anticipative policy for a given `scenario`. 
Returns the optimal action `one_hot_argmax(scenario)`.
"""
function (::AnticipativeSolver)(scenario; context...)
    return one_hot_argmax(scenario)
end

"""
$TYPEDSIGNATURES

Evaluate the anticipative policy with a parametric prediction `θ` and a `scenario`.
Returns the optimal action for the combined signal `one_hot_argmax(scenario + θ)`.
"""
function (::AnticipativeSolver)(θ, scenario; context...)
    ξ = scenario + θ
    return one_hot_argmax(ξ)
end
