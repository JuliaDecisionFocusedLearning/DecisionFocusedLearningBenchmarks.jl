module DFLBenchmarksPlotsExt

using DecisionFocusedLearningBenchmarks
using DocStringExtensions: TYPEDSIGNATURES
using LaTeXStrings: @L_str
using Plots
import DecisionFocusedLearningBenchmarks:
    has_visualization, plot_instance, plot_solution, plot_trajectory, animate_trajectory

include("plots/argmax2d_plots.jl")
include("plots/warcraft_plots.jl")
include("plots/svs_plots.jl")
include("plots/dvs_plots.jl")

"""
    plot_solution(bench::AbstractBenchmark, sample::DataSample, y; kwargs...)

Reconstruct a new sample with `y` overridden and delegate to the 2-arg
[`plot_solution`](@ref). Only available when `Plots` is loaded.
"""
function plot_solution(bench::AbstractBenchmark, sample::DataSample, y; kwargs...)
    return plot_solution(
        bench,
        DataSample(;
            sample.instance_kwargs..., x=sample.x, θ=sample.θ, y=y, extra=sample.extra
        );
        kwargs...,
    )
end

end
