"""
$TYPEDSIGNATURES

SAA baseline policy: builds a stochastic instance from all K scenarios and solves
via column generation.
Returns a single labeled [`DataSample`](@ref) with `extra=(; scenarios)`.
"""
function svs_saa_policy(sample, scenarios)
    stochastic_inst = build_stochastic_instance(sample.instance, scenarios)
    y = column_generation_algorithm(stochastic_inst)
    return [DataSample(; sample.context..., x=sample.x, y, extra=(; scenarios))]
end

"""
$TYPEDSIGNATURES

Deterministic baseline policy: solves the deterministic MIP (ignores scenario delays).
Returns a single labeled [`DataSample`](@ref) with `extra=(; scenarios)`.
"""
function svs_deterministic_policy(sample, scenarios; model_builder=highs_model)
    y = deterministic_mip(sample.instance; model_builder)
    return [DataSample(; sample.context..., x=sample.x, y, extra=(; scenarios))]
end

"""
$TYPEDSIGNATURES

Local search baseline policy: builds a stochastic instance from all K scenarios and
solves via local search heuristic.
Returns a single labeled [`DataSample`](@ref) with `extra=(; scenarios)`.
"""
function svs_local_search_policy(sample, scenarios)
    stochastic_inst = build_stochastic_instance(sample.instance, scenarios)
    y = local_search(stochastic_inst)
    return [DataSample(; sample.context..., x=sample.x, y, extra=(; scenarios))]
end

"""
$TYPEDSIGNATURES

Return the named baseline policies for [`StochasticVehicleSchedulingBenchmark`](@ref).
Each policy has signature `(sample, scenarios) -> Vector{DataSample}`.
"""
function svs_generate_baseline_policies(::StochasticVehicleSchedulingBenchmark)
    return (;
        deterministic=Policy(
            "Deterministic MIP", "Ignores delays", svs_deterministic_policy
        ),
        saa=Policy("SAA (col gen)", "Stochastic MIP over K scenarios", svs_saa_policy),
        local_search=Policy(
            "Local search", "Heuristic with K scenarios", svs_local_search_policy
        ),
    )
end
