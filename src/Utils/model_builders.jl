"""
$TYPEDSIGNATURES

Initialize a HiGHS model (with disabled logging).
"""
function highs_model()
    model = Model(HiGHS.Optimizer)
    return model
end

"""
$TYPEDSIGNATURES

Initialize a SCIP model (with disabled logging).
"""
function scip_model()
    model = Model(SCIP.Optimizer)
    return model
end
