module Warcraft

using ..Utils

using DataDeps
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
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

include("dataset.jl")

"""
$TYPEDSIGNATURES

Plot the image `im`, the weights `weights`, and the path `path` on the same Figure.
"""
function Utils.plot_data(
    ::WarcraftBenchmark, x, y, θ; θ_title="Weights", y_title="Path", θ_true=θ
)
    im = dropdims(x; dims=4)
    img = convert_image_for_plot(im)
    p1 = Plots.plot(
        img; aspect_ratio=:equal, framestyle=:none, size=(300, 300), title="Terrain image"
    )
    p2 = Plots.heatmap(
        abs.(θ);
        yflip=true,
        aspect_ratio=:equal,
        framestyle=:none,
        padding=(0.0, 0.0),
        size=(300, 300),
        legend=false,
        title=θ_title,
        clim=(minimum(θ_true), maximum(θ_true)),
    )
    p3 = Plots.plot(
        Gray.(y .* 0.7);
        aspect_ratio=:equal,
        framestyle=:none,
        size=(300, 300),
        title=y_title,
    )
    return plot(p1, p2, p3; layout=(1, 3), size=(900, 300))
end

"""
$TYPEDSIGNATURES
"""
function Utils.generate_dataset(::WarcraftBenchmark, dataset_size::Int=10)
    decompressed_path = datadep"warcraft/data"
    return create_dataset(decompressed_path, dataset_size)
end

"""
$TYPEDSIGNATURES
"""
function Utils.generate_maximizer(::WarcraftBenchmark; dijkstra=true)
    return dijkstra ? dijkstra_maximizer : bellman_maximizer
end

"""
$TYPEDSIGNATURES

Create and return a `Flux.Chain` embedding for the Warcraft terrains, inspired by [differentiation of blackbox combinatorial solvers](https://github.com/martius-lab/blackbox-differentiation-combinatorial-solvers/blob/master/models.py).

The embedding is made as follows:
1) The first 5 layers of ResNet18 (convolution, batch normalization, relu, maxpooling and first resnet block).
2) An adaptive maxpooling layer to get a (12x12x64) tensor per input image.
3) An average over the third axis (of size 64) to get a (12x12x1) tensor per input image.
4) The element-wize `neg_tensor` function to get cell weights of proper sign to apply shortest path algorithms.
5) A squeeze function to forget the two last dimensions.
"""
function Utils.generate_statistical_model(::WarcraftBenchmark)
    Random.seed!(67)
    resnet18 = ResNet(18; pretrain=false, nclasses=1)
    model_embedding = Chain(
        resnet18.layers[1][1][1],
        resnet18.layers[1][1][2],
        resnet18.layers[1][1][3],
        resnet18.layers[1][2][1],
        AdaptiveMaxPool((12, 12)),
        average_tensor,
        neg_tensor,
        squeeze_last_dims,
    )
    return model_embedding
end

export WarcraftBenchmark,
    generate_dataset, generate_maximizer, generate_statistical_model, train_test_split
export plot_data, plot_image_path

end
