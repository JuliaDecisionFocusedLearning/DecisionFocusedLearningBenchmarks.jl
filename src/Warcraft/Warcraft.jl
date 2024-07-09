module Warcraft

using ..Utils

using DataDeps
using Flux
using Graphs
using Images
using LinearAlgebra
using Metalhead
using NPZ
using Plots
using Random
using SimpleWeightedGraphs
using SparseArrays

include("grid_graph.jl")

export warcraft_grid_graph, index_to_coord, coord_to_index
export WarcraftBenchmark,
    generate_dataset, generate_maximizer, generate_statistical_model, train_test_split
export plot_data, plot_image_path

end
