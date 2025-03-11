module StochasticVehicleScheduling

export StochasticVehicleSchedulingBenchmark
export generate_dataset, generate_maximizer, generate_statistical_model
export plot_instance, plot_solution
export compact_linearized_mip, compact_mip, column_generation_algorithm, local_search
export evaluate_solution, is_feasible

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using ConstrainedShortestPaths:
    stochastic_routing_shortest_path, stochastic_routing_shortest_path_with_threshold
using Distributions: Distribution, LogNormal, Uniform, DiscreteUniform
using Flux: Chain, Dense
using Graphs:
    AbstractGraph,
    SimpleDiGraph,
    add_edge!,
    nv,
    ne,
    edges,
    src,
    dst,
    has_edge,
    inneighbors,
    outneighbors
using JuMP:
    JuMP, Model, @variable, @objective, @constraint, optimize!, value, set_silent, dual
using Plots: Plots, plot, plot!, scatter!, annotate!, text
using Printf: @printf
using Random: Random, AbstractRNG, MersenneTwister
using SparseArrays: sparse
using Statistics: quantile, mean

include("utils.jl")
include("instance/constants.jl")
include("instance/task.jl")
include("instance/district.jl")
include("instance/city.jl")
include("instance/features.jl")
include("instance/instance.jl")

include("solution/solution.jl")
include("solution/algorithms/mip.jl")
include("solution/algorithms/column_generation.jl")
include("solution/algorithms/local_search.jl")

include("maximizer.jl")

"""
$TYPEDEF

Data structure for a stochastic vehicle scheduling benchmark.

# Fields
$TYPEDFIELDS
"""
@kwdef struct StochasticVehicleSchedulingBenchmark <: AbstractBenchmark
    "number of tasks in each instance"
    nb_tasks::Int = 25
    "number of scenarios in each instance"
    nb_scenarios::Int = 10
end

function Utils.objective_value(
    ::StochasticVehicleSchedulingBenchmark, sample::DataSample, y::BitVector
)
    return evaluate_solution(y, sample.instance)
end

"""
$TYPEDSIGNATURES

Create a dataset of `dataset_size` instances for the given `StochasticVehicleSchedulingBenchmark`.
If you want to not add label solutions in the dataset, set `compute_solutions=false`.
By default, they will be computed using column generation.
Note that computing solutions can be time-consuming, especially for large instances.
You can also use instead `compact_mip` or `compact_linearized_mip` as the algorithm to compute solutions.
If you want to provide a custom algorithm to compute solutions, you can pass it as the `algorithm` keyword argument.
If `algorithm` takes keyword arguments, you can pass them as well directly in `kwargs...`.
If `store_city=false`, the coordinates and unnecessary information about instances will not be stored in the dataset.
"""
function Utils.generate_dataset(
    benchmark::StochasticVehicleSchedulingBenchmark,
    dataset_size::Int;
    compute_solutions=true,
    seed=nothing,
    rng=MersenneTwister(0),
    algorithm=column_generation_algorithm,
    store_city=true,
    kwargs...,
)
    (; nb_tasks, nb_scenarios) = benchmark
    Random.seed!(rng, seed)
    instances = [
        Instance(; nb_tasks, nb_scenarios, rng, store_city) for _ in 1:dataset_size
    ]
    features = get_features.(instances)
    if compute_solutions
        solutions = [algorithm(instance; kwargs...).value for instance in instances]
        return [
            DataSample(; x=feature, instance, y_true=solution) for
            (instance, feature, solution) in zip(instances, features, solutions)
        ]
    end
    # else
    return [
        DataSample(; x=feature, instance) for
        (instance, feature) in zip(instances, features)
    ]
end

"""
$TYPEDEF

Deterministic vsp maximizer for the [StochasticVehicleSchedulingBenchmark](@ref).
"""
struct StochasticVechicleSchedulingMaximizer{M}
    "mip solver model to use"
    model_builder::M
end

"""
$TYPEDSIGNATURES
"""
function Utils.generate_maximizer(
    bench::StochasticVehicleSchedulingBenchmark; model_builder=highs_model
)
    return StochasticVechicleSchedulingMaximizer(model_builder)
end

"""
$TYPEDSIGNATURES

Apply the maximizer with the stored model builder.
"""
function (maximizer::StochasticVechicleSchedulingMaximizer)(
    θ::AbstractVector; instance::Instance, kwargs...
)
    return vsp_maximizer(θ; instance, model_builder=maximizer.model_builder, kwargs...)
end

"""
$TYPEDSIGNATURES
"""
function Utils.generate_statistical_model(
    ::StochasticVehicleSchedulingBenchmark; seed=nothing
)
    Random.seed!(seed)
    return Chain(Dense(20 => 1; bias=false), vec)
end

"""
$TYPEDSIGNATURES
"""
function plot_instance(
    ::StochasticVehicleSchedulingBenchmark,
    sample::DataSample{<:Instance{City}};
    color_scheme=:lightrainbow,
    kwargs...,
)
    (; tasks, district_width, width) = sample.instance.city
    ticks = 0:district_width:width
    max_time = maximum(t.end_time for t in sample.instance.city.tasks[1:(end - 1)])
    fig = plot(;
        xlabel="x",
        ylabel="y",
        gridlinewidth=3,
        aspect_ratio=:equal,
        size=(500, 500),
        xticks=ticks,
        yticks=ticks,
        xlims=(-1, width + 1),
        ylims=(-1, width + 1),
        clim=(0.0, max_time),
        label=nothing,
        colorbar_title="Time",
    )
    scatter!(
        fig,
        [tasks[1].start_point.x],
        [tasks[1].start_point.y];
        label=nothing,
        marker=:rect,
        markersize=10,
    )
    annotate!(fig, (tasks[1].start_point.x, tasks[1].start_point.y, text("0", 10)))
    for (i_task, task) in enumerate(tasks[2:(end - 1)])
        (; start_point, end_point) = task
        points = [(start_point.x, start_point.y), (end_point.x, end_point.y)]
        plot!(fig, points; color=:black, label=nothing)
        scatter!(
            fig,
            points[1];
            markersize=10,
            marker=:rect,
            marker_z=task.start_time,
            colormap=:turbo,
            label=nothing,
        )
        scatter!(
            fig,
            points[2];
            markersize=10,
            marker=:rect,
            marker_z=task.end_time,
            colormap=:turbo,
            label=nothing,
            # color=palette[max(floor(Int, task.end_time), 1)],
        )
        annotate!(fig, (points[1]..., text("$(i_task)", 10)))
    end
    return fig
end

"""
$TYPEDSIGNATURES
"""
function plot_solution(
    ::StochasticVehicleSchedulingBenchmark, sample::DataSample{<:Instance{City}}; kwargs...
)
    (; tasks, district_width, width) = sample.instance.city
    ticks = 0:district_width:width
    solution = Solution(sample.y_true, sample.instance)
    path_list = compute_path_list(solution)
    fig = plot(;
        xlabel="x",
        ylabel="y",
        legend=false,
        gridlinewidth=3,
        aspect_ratio=:equal,
        size=(500, 500),
        xticks=ticks,
        yticks=ticks,
        xlims=(-1, width + 1),
        ylims=(-1, width + 1),
    )
    for path in path_list
        X = Float64[]
        Y = Float64[]
        (; start_point, end_point) = tasks[path[1]]
        (; x, y) = end_point
        push!(X, x)
        push!(Y, y)
        for task in path[2:end]
            (; start_point, end_point) = tasks[task]
            push!(X, start_point.x)
            push!(Y, start_point.y)
            push!(X, end_point.x)
            push!(Y, end_point.y)
        end
        plot!(fig, X, Y; marker=:circle)
    end
    return fig
end

end
