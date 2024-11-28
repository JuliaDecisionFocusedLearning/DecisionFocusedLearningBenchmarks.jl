"""
$TYPEDEF

Data structure for a city in the vehicle scheduling problem.
Contains all the relevant information to build an instance of the problem.

# Fields
$TYPEDFIELDS
"""
struct City
    "city width (in minutes)"
    width::Int
    # Objectives ponderation
    "cost of a vehicle in the objective function"
    vehicle_cost::Float64
    "cost of one minute delay in the objective function"
    delay_cost::Float64
    # Tasks
    "number of tasks to fulfill"
    nb_tasks::Int
    "tasks list (see [`Task`](@ref)), that should be ordered by start time"
    tasks::Vector{Task}
    # Stochastic specific stuff
    "idth (in minutes) of each district"
    district_width::Int
    "districts matrix (see [`District`](@ref)), indices corresponding to their relative positions"
    districts::Matrix{District}
    "a log-normal distribution modeling delay between districts"
    random_inter_area_factor::LogNormal{Float64}
    "size (nb_scenarios, 24), each row correspond to one scenario, each column to one hour of the day"
    scenario_inter_area_factor::Matrix{Float64}
end

"""
$TYPEDSIGNATURES

Constructor for [`City`](@ref).
"""
function City(;
    nb_scenarios=default_nb_scenarios,
    width=default_width,
    vehicle_cost=default_vehicle_cost,
    nb_tasks=default_nb_tasks,
    tasks=Vector{Task}(undef, nb_tasks + 2),
    district_width=default_district_width,
    districts=Matrix{District}(undef, width ÷ district_width, width ÷ district_width),
    delay_cost=default_delay_cost,
    random_inter_area_factor=default_random_inter_area_factor,
    scenario_inter_area_factor=zeros(nb_scenarios, 24),
)
    return City(
        width,
        vehicle_cost,
        delay_cost,
        nb_tasks,
        tasks,
        district_width,
        districts,
        random_inter_area_factor,
        scenario_inter_area_factor,
    )
end

"""
$TYPEDSIGNATURES

- Creates a city from `city_kwargs`
- Depot location at city center
- Randomize tasks, and add two dummy tasks : one `source` task at time=0 from the depot,
    and one `destination` task ending at time=end at depot
- Roll every scenario.
"""
function create_random_city(;  # TODO: use an rng here
    αᵥ_low=default_αᵥ_low,
    αᵥ_high=default_αᵥ_high,
    first_begin_time=default_first_begin_time,
    last_begin_time=default_last_begin_time,
    district_μ=default_district_μ,
    district_σ=default_district_σ,
    task_μ=default_task_μ,
    task_σ=default_task_σ,
    seed=nothing,
    rng=MersenneTwister(0),
    city_kwargs...,
)
    Random.seed!(rng, seed)
    city = City(; city_kwargs...)
    init_districts!(city, district_μ, district_σ; rng=rng)
    init_tasks!(
        city, αᵥ_low, αᵥ_high, first_begin_time, last_begin_time, task_μ, task_σ; rng=rng
    )
    generate_scenarios!(city; rng=rng)
    compute_perturbed_end_times!(city)
    return city
end

"""
$TYPEDSIGNATURES

Initialize the districts of the city.
"""
function init_districts!(
    city::City, district_μ::Distribution, district_σ::Distribution; rng::AbstractRNG
)
    nb_scenarios = size(city.scenario_inter_area_factor, 1)
    nb_district_per_edge = city.width ÷ city.district_width
    for x in 1:nb_district_per_edge
        for y in 1:nb_district_per_edge
            μ = rand(rng, district_μ)
            σ = rand(rng, district_σ)
            city.districts[x, y] = District(;
                random_delay=LogNormal(μ, σ), nb_scenarios=nb_scenarios
            )
        end
    end
    return nothing
end

"""
$TYPEDSIGNATURES

Draw the tasks of the city.
"""
function init_tasks!(
    city::City,
    αᵥ_low::Real,
    αᵥ_high::Real,
    first_begin_time::Real,
    last_begin_time::Real,
    task_μ::Distribution,
    task_σ::Distribution;
    rng::AbstractRNG,
)
    nb_scenarios = size(city.scenario_inter_area_factor, 1)

    point_distribution = Uniform(0, city.width)
    start_time_distribution = Uniform(first_begin_time, last_begin_time)
    travel_time_multiplier_distribution = Uniform(αᵥ_low, αᵥ_high)

    for i_task in 1:(city.nb_tasks)
        start_point = draw_random_point(point_distribution; rng=rng)
        end_point = draw_random_point(point_distribution; rng=rng)

        start_time = rand(rng, start_time_distribution)
        end_time =
            start_time +
            rand(rng, travel_time_multiplier_distribution) *
            distance(start_point, end_point)

        μ = rand(rng, task_μ)
        σ = rand(rng, task_σ)
        random_delay = LogNormal(μ, σ)

        city.tasks[i_task + 1] = Task(;
            type=job::TaskType,
            start_point=start_point,
            end_point=end_point,
            start_time=start_time,
            end_time=end_time,
            random_delay=random_delay,
            nb_scenarios=nb_scenarios,
        )
    end

    # add start and final "artificial" tasks
    city_center = Point(city.width / 2, city.width / 2)  # ? hard coded ?
    city.tasks[1] = Task(;
        type=depot_start::TaskType,
        start_point=city_center,
        end_point=city_center,
        start_time=0.0,
        end_time=0.0,
        random_delay=ZERO_UNIFORM,
        nb_scenarios=nb_scenarios,
    )
    final_task_time = 24 * 60.0 # ? hard coded ?
    city.tasks[end] = Task(;
        type=depot_end::TaskType,
        start_point=city_center,
        end_point=city_center,
        start_time=final_task_time,
        end_time=final_task_time,
        random_delay=ZERO_UNIFORM,
        nb_scenarios=nb_scenarios,
    )

    # sort tasks by start time
    sort!(city.tasks; by=task -> task.start_time, rev=false)
    return nothing
end

"""
    get_district(point::Point, city::City)

Return indices of the `city` district containing `point`.
"""
function get_district(point::Point, city::City)
    return trunc(Int, point.x / city.district_width) + 1,
    trunc(Int, point.y / city.district_width) + 1
end

"""
$TYPEDSIGNATURES

Draw all delay scenarios for the city.
"""
function generate_scenarios!(city::City; rng::AbstractRNG)
    # roll all tasks
    for task in city.tasks
        roll(task, rng)
    end

    # roll all districts
    for district in city.districts
        roll(district, rng)
    end

    # roll inter-district
    nb_scenarios, nb_hours = size(city.scenario_inter_area_factor)
    for s in 1:nb_scenarios
        previous_delay = 0.0
        for h in 1:nb_hours
            previous_delay =
                (previous_delay + 0.1) * rand(rng, city.random_inter_area_factor)
            city.scenario_inter_area_factor[s, h] = previous_delay
        end
    end
    return nothing
end

"""
$TYPEDSIGNATURES

Compute the end times of the tasks for each scenario.
"""
function compute_perturbed_end_times!(city::City)
    nb_scenarios = size(city.scenario_inter_area_factor, 1)

    for task in city.tasks[2:(end - 1)]
        start_time = task.start_time
        end_time = task.end_time
        start_point = task.start_point
        end_point = task.end_point

        origin_x, origin_y = get_district(start_point, city)
        destination_x, destination_y = get_district(end_point, city)
        origin_district = city.districts[origin_x, origin_y]
        destination_district = city.districts[destination_x, destination_y]

        scenario_start_time = task.scenario_start_time
        origin_delay = origin_district.scenario_delay
        destination_delay = destination_district.scenario_delay
        inter_area_delay = city.scenario_inter_area_factor

        for s in 1:nb_scenarios
            ξ₁ = scenario_start_time[s]
            ξ₂ = ξ₁ + origin_delay[s, hour_of(ξ₁)]
            ξ₃ = ξ₂ + end_time - start_time + inter_area_delay[s, hour_of(ξ₂)]
            task.scenario_end_time[s] = ξ₃ + destination_delay[s, hour_of(ξ₃)]
        end
    end
    return nothing
end
