"""
$TYPEDEF

Data structure for a district in the vehicle scheduling problem.

# Fields
$TYPEDFIELDS
"""
struct District
    "log-normal distribution modeling the district delay"
    random_delay::LogNormal{Float64}
    "size (nb_scenarios, 24), observed delays for each scenario and hour of the day"
    scenario_delay::Matrix{Float64}
end

"""
$TYPEDSIGNATURES

Constructor for [`District`](@ref).
Initialize a district with a given number of scenarios, with zeros in `scenario_delay`.
"""
function District(; random_delay::LogNormal{Float64}, nb_scenarios::Int)
    return District(random_delay, zeros(nb_scenarios, 24))
end

"""
$TYPEDSIGNATURES

Return one scenario of future delay given current delay and delay distribution.
"""
function scenario_next_delay(
    previous_delay::Real, random_delay::Distribution, rng::AbstractRNG
)
    return previous_delay / 2.0 + rand(rng, random_delay)
end

"""
$TYPEDSIGNATURES

Populate `scenario_delay` with delays drawn from `random_delay` distribution
for each (scenario, hour) pair.
"""
function roll(district::District, rng::AbstractRNG)
    nb_scenarios, nb_hours = size(district.scenario_delay)
    # Loop on scenarios
    for s in 1:nb_scenarios
        previous_delay = 0.0
        # Loop on hours
        for h in 1:nb_hours
            previous_delay = scenario_next_delay(previous_delay, district.random_delay, rng)
            district.scenario_delay[s, h] = previous_delay
        end
    end
    return nothing
end
