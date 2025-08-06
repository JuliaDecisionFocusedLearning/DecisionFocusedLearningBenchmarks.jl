using DocStringExtensions: TYPEDSIGNATURES
using JuMP: Model

@info "Creating a GRB_ENV const for DecisionFocusedLearningBenchmarks..."
# Gurobi package setup (see https://github.com/jump-dev/Gurobi.jl/issues/424)
const GRB_ENV = Ref{Gurobi.Env}()
GRB_ENV[] = Gurobi.Env()
export GRB_ENV

"""
$TYPEDSIGNATURES

Create an empty Gurobi model.
"""
function grb_model()
    model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
    return model
end

export grb_model
