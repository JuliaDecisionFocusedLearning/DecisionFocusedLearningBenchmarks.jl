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

include("DFLBenchmarkPlotsExt/argmax_plots.jl")
include("DFLBenchmarkPlotsExt/argmax2d_plots.jl")
include("DFLBenchmarkPlotsExt/ranking_plots.jl")
include("DFLBenchmarkPlotsExt/subset_selection_plots.jl")
include("DFLBenchmarkPlotsExt/portfolio_plots.jl")
include("DFLBenchmarkPlotsExt/shortest_path_plots.jl")
include("DFLBenchmarkPlotsExt/contextual_stochastic_argmax_plots.jl")
include("DFLBenchmarkPlotsExt/warcraft_plots.jl")
include("DFLBenchmarkPlotsExt/svs_plots.jl")
include("DFLBenchmarkPlotsExt/dvs_plots.jl")
include("DFLBenchmarkPlotsExt/dynamic_assortment_plots.jl")
include("DFLBenchmarkPlotsExt/maintenance_plots.jl")

end
