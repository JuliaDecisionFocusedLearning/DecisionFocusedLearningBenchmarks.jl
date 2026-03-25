"""
$TYPEDSIGNATURES

SAA baseline policy: builds a stochastic instance from all K scenarios and solves
via column generation.
Returns a single labeled [`DataSample`](@ref) with `extra=(; scenarios)`.
"""
function svs_saa_policy(ctx_sample, scenarios)
    stochastic_inst = build_stochastic_instance(ctx_sample.instance, scenarios)
    y = column_generation_algorithm(stochastic_inst)
    return [
        DataSample(;
            ctx_sample.maximizer_kwargs...,
            x=ctx_sample.x,
            y,
            extra=(; ctx_sample.extra..., scenarios),
        ),
    ]
end

"""
$TYPEDSIGNATURES

Deterministic baseline policy: solves the deterministic MIP (ignores scenario delays).
Returns a single labeled [`DataSample`](@ref) with `extra=(; scenarios)`.
"""
function svs_deterministic_policy(ctx_sample, scenarios; model_builder=highs_model)
    y = deterministic_mip(ctx_sample.instance; model_builder)
    return [
        DataSample(;
            ctx_sample.maximizer_kwargs...,
            x=ctx_sample.x,
            y,
            extra=(; ctx_sample.extra..., scenarios),
        ),
    ]
end

"""
$TYPEDSIGNATURES

Local search baseline policy: builds a stochastic instance from all K scenarios and
solves via local search heuristic.
Returns a single labeled [`DataSample`](@ref) with `extra=(; scenarios)`.
"""
function svs_local_search_policy(ctx_sample, scenarios)
    stochastic_inst = build_stochastic_instance(ctx_sample.instance, scenarios)
    y = local_search(stochastic_inst)
    return [
        DataSample(;
            ctx_sample.maximizer_kwargs...,
            x=ctx_sample.x,
            y,
            extra=(; ctx_sample.extra..., scenarios),
        ),
    ]
end

"""
$TYPEDSIGNATURES

Exact SAA MIP policy (linearized): solves the stochastic VSP exactly for the given
scenarios via [`compact_linearized_mip`](@ref).
Returns a single labeled [`DataSample`](@ref) with `extra=(; scenarios)`.

Prefer this over [`svs_saa_policy`](@ref) when an exact solution is needed; requires
SCIP (default) or Gurobi.
"""
function svs_saa_mip_policy(ctx_sample, scenarios; model_builder=scip_model)
    y = compact_linearized_mip(ctx_sample.instance, scenarios; model_builder)
    return [
        DataSample(;
            ctx_sample.maximizer_kwargs...,
            x=ctx_sample.x,
            y,
            extra=(; ctx_sample.extra..., scenarios),
        ),
    ]
end

"""
$TYPEDSIGNATURES

Return the named baseline policies for [`StochasticVehicleSchedulingBenchmark`](@ref).
Each policy has signature `(ctx_sample, scenarios) -> Vector{DataSample}`.
"""
function svs_generate_baseline_policies(::StochasticVehicleSchedulingBenchmark)
    return (;
        deterministic=Policy(
            "Deterministic MIP", "Ignores delays", svs_deterministic_policy
        ),
        saa=Policy("SAA (col gen)", "Stochastic MIP over K scenarios", svs_saa_policy),
        saa_mip=Policy(
            "SAA (exact MIP)",
            "Exact stochastic MIP over K scenarios via compact linearized formulation",
            svs_saa_mip_policy,
        ),
        local_search=Policy(
            "Local search", "Heuristic with K scenarios", svs_local_search_policy
        ),
    )
end
