module DFLBenchmarksPlotsExt

using DecisionFocusedLearningBenchmarks
using DocStringExtensions: TYPEDSIGNATURES
using LaTeXStrings: @L_str
using Plots
import DecisionFocusedLearningBenchmarks:
    has_visualization, plot_instance, plot_sample, plot_trajectory, animate_trajectory

include("plots/argmax_plots.jl")
include("plots/argmax2d_plots.jl")
include("plots/ranking_plots.jl")
include("plots/subset_selection_plots.jl")
include("plots/portfolio_plots.jl")
include("plots/shortest_path_plots.jl")
include("plots/contextual_stochastic_argmax_plots.jl")
include("plots/warcraft_plots.jl")
include("plots/svs_plots.jl")
include("plots/dvs_plots.jl")
include("plots/dynamic_assortment_plots.jl")
include("plots/maintenance_plots.jl")

"""
    plot_sample(bench::AbstractBenchmark, sample::DataSample, y; kwargs...)

Reconstruct a new sample with `y` overridden and delegate to the 2-arg
[`plot_sample`](@ref). Only available when `Plots` is loaded.
"""
function plot_sample(bench::AbstractBenchmark, sample::DataSample, y; kwargs...)
    return plot_sample(
        bench,
        DataSample(; sample.context..., x=sample.x, θ=sample.θ, y=y, extra=sample.extra);
        kwargs...,
    )
end

end
