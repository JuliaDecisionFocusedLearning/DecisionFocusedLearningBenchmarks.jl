"""
$TYPEDEF

Anticipative policy for the Dynamic Vehicle Scheduling Problem.
"""
struct AnticipativeVSPPolicy <: AbstractDynamicPolicy end

"""
$TYPEDSIGNATURES

Apply the anticipative policy to the environment.
"""
function run_policy!(
    ::AnticipativeVSPPolicy, env::DVSPEnv, scenario=env.scenario; model_builder=highs_model
)
    return anticipative_solver(env, scenario; model_builder, reset_env=true)
end
