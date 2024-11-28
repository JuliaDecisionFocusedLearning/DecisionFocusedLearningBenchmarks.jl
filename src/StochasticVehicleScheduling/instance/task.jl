@enum TaskType depot_start job depot_end

"""
$TYPEDEF

Data structure for a task in the vehicle scheduling problem.

# Fields
$TYPEDFIELDS
"""
struct Task
    "type of the task (depot start, job, or depot end)"
    type::TaskType
    "starting location of the task"
    start_point::Point
    "end location of the task"
    end_point::Point
    "start time (in minutes) of the task"
    start_time::Float64
    "end time (in minutes) of the task"
    end_time::Float64
    "lognormal distribution modeling the task start delay"
    random_delay::LogNormal{Float64}
    "size (nb_scenarios), observed delayed start times for each scenario"
    scenario_start_time::Vector{Float64}
    "size (nb_scenarios), observed delayed end times for each scenario"
    scenario_end_time::Vector{Float64}
end

"""
$TYPEDSIGNATURES

Constructor for [`Task`](@ref).
"""
function Task(;
    type::TaskType=job,
    start_point::Point,
    end_point::Point,
    start_time::Float64,
    end_time::Float64,
    nb_scenarios::Int,
    random_delay::LogNormal{Float64}=ZERO_UNIFORM,
)
    return Task(
        type,
        start_point,
        end_point,
        start_time,
        end_time,
        random_delay,
        zeros(nb_scenarios) .+ start_time,
        zeros(nb_scenarios) .+ end_time,
    )
end

"""
$TYPEDSIGNATURES

Return the number of scenarios for the given task.
"""
function nb_scenarios(task::Task)
    return length(task.scenario_start_time)
end

"""
$TYPEDSIGNATURES

Populate `scenario_start_time` with delays drawn from the `random_delay` distribution of
the given task for each scenario.
"""
function roll(task::Task, rng::AbstractRNG)
    S = nb_scenarios(task)
    task.scenario_start_time .= task.start_time .+ rand(rng, task.random_delay, S)
    return nothing
end

function Base.show(io::IO, task::Task)
    @printf(
        "(%.2f, %.2f) -> (%.2f, %.2f), [%.2f, %.2f], %s, %s, %s",
        task.start_point.x,
        task.start_point.y,
        task.end_point.x,
        task.end_point.y,
        task.start_time,
        task.end_time,
        task.type,
        task.scenario_start_time,
        task.scenario_end_time,
    )
end
