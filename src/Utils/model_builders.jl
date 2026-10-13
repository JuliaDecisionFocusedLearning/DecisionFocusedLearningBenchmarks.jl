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
    # Accept partial primal starts however few variables they fix (default 0.85)
    set_attribute(model, "heuristics/completesol/maxunknownrate", 1.0)
    return model
end
