# --- Default values for structure attributes ---

const ZERO_UNIFORM = LogNormal(-Inf, 1.0)  # always returns 0 (useful for type stability reasons)

const default_delay_cost = 2.0             # cost of one minute of delay
const default_vehicle_cost = 1000.0        # cost of one vehicle
const default_width = 50                   # width (in minutes) of the squared city
const default_αᵥ_low = 1.2                 # used for drawing random tasks
const default_αᵥ_high = 1.6                # used for drawing random tasks
const default_first_begin_time = 60.0 * 6  # Start of time window at 6AM
const default_last_begin_time = 60.0 * 20  # End of time window at 8PM

const default_district_width = 10          # width (in minutes) of each squared district
const default_random_inter_area_factor = LogNormal(0.02, 0.05)

const default_district_μ = Uniform(0.8, 1.2)
const default_district_σ = Uniform(0.4, 0.6)

const default_task_μ = Uniform(1.0, 3.0)
const default_task_σ = Uniform(0.8, 1.2)

const default_nb_tasks = 10               # Number of tasks in an instnace
const default_nb_scenarios = 1            # Number of scenrios in an instance
