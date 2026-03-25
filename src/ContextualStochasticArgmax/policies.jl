using Statistics: mean

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
            ctx_sample.maximizer_kwargs...,
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
