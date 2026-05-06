module DFLBenchmarksPlotsExt

using DecisionFocusedLearningBenchmarks
using DocStringExtensions: TYPEDSIGNATURES
using LaTeXStrings: @L_str
using Plots
import DecisionFocusedLearningBenchmarks:
    has_visualization, plot_context, plot_sample, plot_trajectory, animate_trajectory

function _step_str(sample::DataSample)
    return hasproperty(sample, :step) ? " (step $(sample.step))" : ""
end

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

end
