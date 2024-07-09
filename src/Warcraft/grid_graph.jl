"""
    warcraft_grid_graph(costs::AbstractMatrix; acyclic::Bool=false)

Convert a grid of Warcraft cell costs into a weighted directed graph from [SimpleWeightedGraphs.jl](https://github.com/JuliaGraphs/SimpleWeightedGraphs.jl), where the vertices correspond to the cells and the edges are weighted by the cost of the arrival cell.

This represents the Warcraft shortest paths problem of

> [Differentiation of Blackbox Combinatorial Solvers](https://openreview.net/forum?id=BkevoJSYPB), Vlastelica et al. (2019)

- If `acyclic = false`, a cell has edges to each one of its 8 neighbors.
- If `acyclic = true`, a cell has edges to its south, east and southeast neighbors only (ensures an acyclic graph where topological sort will work)
"""
function warcraft_grid_graph(costs::AbstractMatrix{R}; acyclic::Bool=false) where {R}
    h, w = size(costs)
    V = h * w
    E = count_edges(h, w; acyclic)

    sources = Int[]
    destinations = Int[]
    weights = R[]

    sizehint!(sources, E)
    sizehint!(destinations, E)
    sizehint!(weights, E)

    for v1 in 1:V
        i1, j1 = index_to_coord(v1, h, w)
        for Δi in (-1, 0, 1), Δj in (-1, 0, 1)
            i2, j2 = i1 + Δi, j1 + Δj
            valid_destination = 1 <= i2 <= h && 1 <= j2 <= w
            valid_step = if acyclic
                (Δi != 0 || Δj != 0) && Δi >= 0 && Δj >= 0
            else
                (Δi != 0 || Δj != 0)
            end
            if valid_destination && valid_step
                v2 = coord_to_index(i2, j2, h, w)
                push!(sources, v1)
                push!(destinations, v2)
                push!(weights, costs[v2])
            end
        end
    end

    return SimpleWeightedDiGraph(sources, destinations, weights)
end

function count_edges(h::Integer, w::Integer; acyclic::Bool)
    @assert h >= 2 && w >= 2
    if acyclic
        return (h - 1) * (w - 1) * 3 + ((h - 1) + (w - 1)) * 1 + 0
    else
        return (h - 2) * (w - 2) * 8 + (2(h - 2) + 2(w - 2)) * 5 + 4 * 3
    end
end

function possible_neighbors(i::Integer, j::Integer)
    return (
        # col - 1
        (i - 1, j - 1),
        (i + 0, j - 1),
        (i + 1, j - 1),
        # col 0
        (i - 1, j + 0),
        (i + 1, j + 0),
        # col + 1
        (i - 1, j + 1),
        (i + 0, j + 1),
        (i + 1, j + 1),
    )
end

"""
    coord_to_index(i, j, h, w)

Given a pair of row-column coordinates `(i, j)` on a grid of size `(h, w)`, compute the corresponding vertex index in the graph generated by [`warcraft_grid_graph`](@ref).
"""
function coord_to_index(i::Integer, j::Integer, h::Integer, w::Integer)
    if (1 <= i <= h) && (1 <= j <= w)
        v = (j - 1) * h + (i - 1) + 1  # enumerate column by column
        return v
    else
        return 0
    end
end

"""
    index_to_coord(v, h, w)

Given a vertex index in the graph generated by [`warcraft_grid_graph`](@ref), compute the corresponding row-column coordinates `(i, j)` on a grid of size `(h, w)`.
"""
function index_to_coord(v::Integer, h::Integer, w::Integer)
    if 1 <= v <= h * w
        j = (v - 1) ÷ h + 1
        i = (v - 1) - h * (j - 1) + 1
        return (i, j)
    else
        return (0, 0)
    end
end

function get_path(parents::AbstractVector{<:Integer}, s::Integer, d::Integer)
    path = [d]
    v = d
    while v != s
        v = parents[v]
        pushfirst!(path, v)
    end
    return path
end

"""
    path_to_matrix(g, path::Vector{<:Integer})

Store the shortest `s -> d` path in `g` as an integer matrix of size `height(g) * width(g)`, where entry `(i,j)` counts the number of visits to the associated vertex.
"""
function path_to_matrix(g, path::Vector{<:Integer})
    y = zeros(Int, 12, 12) # ! hardcoded
    for v in path
        i, j = index_to_coord(v, 12, 12)
        y[i, j] += 1
    end
    return y
end

"""
	read_dataset(decompressed_path::String, dtype::String="train")

Read the dataset of type `dtype` at the `decompressed_path` location.
The dataset is made of images of Warcraft terrains, cell cost labels and shortest path labels.
They are returned separately, with proper axis permutation and image scaling to be consistent with 
`Flux` embeddings.
"""
function read_dataset(decompressed_path::String, dtype::String="train")
    # Open files
    data_dir = joinpath(decompressed_path, "warcraft_shortest_path_oneskin", "12x12")
    data_suffix = "maps"
    terrain_images = npzread(joinpath(data_dir, dtype * "_" * data_suffix * ".npy"))
    terrain_weights = npzread(joinpath(data_dir, dtype * "_vertex_weights.npy"))
    terrain_labels = npzread(joinpath(data_dir, dtype * "_shortest_paths.npy"))
    # Reshape for Flux
    terrain_images = permutedims(terrain_images, (2, 3, 4, 1))
    terrain_labels = permutedims(terrain_labels, (2, 3, 1))
    terrain_weights = permutedims(terrain_weights, (2, 3, 1))
    # Normalize images
    terrain_images = Array{Float32}(terrain_images ./ 255)
    return terrain_images, terrain_labels, terrain_weights
end

"""
	create_dataset(decompressed_path::String, nb_samples::Int=10000)

Create the dataset corresponding to the data located at `decompressed_path`, possibly sub-sampling `nb_samples` points.
The dataset is made of images of Warcraft terrains, cell cost labels and shortest path labels.
It is a `Vector` of tuples, each `Tuple` being a dataset point.
"""
function create_dataset(decompressed_path::String, nb_samples::Int=10000)
    terrain_images, terrain_labels, terrain_weights =
        read_dataset(decompressed_path, "train")
    X = [
        reshape(terrain_images[:, :, :, i], (size(terrain_images[:, :, :, i])..., 1))
        for i in 1:nb_samples
    ]
    Y = [terrain_labels[:, :, i] for i in 1:nb_samples]
    WG = [terrain_weights[:, :, i] for i in 1:nb_samples]
    return collect(zip(X, Y, WG))
end

"""
	train_test_split(X::AbstractVector, train_percentage::Real=0.5)

Split a dataset contained in `X` into train and test datasets.
The proportion of the initial dataset kept in the train set is `train_percentage`.
"""
function train_test_split(X::AbstractVector, train_percentage::Real=0.5)
    N = length(X)
    N_train = floor(Int, N * train_percentage)
    N_test = N - N_train
    train_ind, test_ind = 1:N_train, (N_train+1):(N_train+N_test)
    X_train, X_test = X[train_ind], X[test_ind]
    return X_train, X_test
end

"""
    convert_image_for_plot(image::Array{Float32,3})::Array{RGB{N0f8},2}
Convert `image` to the proper data format to enable plots in Julia.
"""
function convert_image_for_plot(image::Array{Float32,3})::Array{RGB{N0f8},2}
    new_img = Array{RGB{N0f8},2}(undef, size(image)[1], size(image)[2])
    for i in 1:size(image)[1]
        for j in 1:size(image)[2]
            new_img[i, j] = RGB{N0f8}(image[i, j, 1], image[i, j, 2], image[i, j, 3])
        end
    end
    return new_img
end

"""
    plot_image_weights_path(;im, weights, path)
Plot the image `im`, the weights `weights`, and the path `path` on the same Figure.
"""
function plot_image_weights_path(x, y, θ; θ_title="Weights", y_title="Path", θ_true=θ)
    im = dropdims(x; dims=4)
    img = convert_image_for_plot(im)
    p1 = Plots.plot(
        img;
        aspect_ratio=:equal,
        framestyle=:none,
        size=(300, 300),
        title="Terrain image",
    )
    p2 = Plots.heatmap(
        θ;
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
    plot_image_path(;im, weights, path)
Plot the image `im`, the weights `weights`, and the path `path` on the same Figure.
"""
function plot_image_path(x, y; y_title="Path")
    im = dropdims(x; dims=4)
    img = convert_image_for_plot(im)
    p1 = Plots.plot(
        img;
        aspect_ratio=:equal,
        framestyle=:none,
        size=(300, 300),
        title="Terrain image",
    )
    p3 = Plots.plot(
        Gray.(y .* 0.7);
        aspect_ratio=:equal,
        framestyle=:none,
        size=(300, 300),
        title=y_title,
    )
    return plot(p1, p3; layout=(1, 2), size=(600, 300))
end

"""
    plot_loss_and_gap(losses::Matrix{Float64}, gaps::Matrix{Float64},  options::NamedTuple; filepath=nothing)

Plot the train and test losses, as well as the train and test gaps computed over epochs.
"""
function plot_loss_and_gap(losses::Matrix{Float64}, gaps::Matrix{Float64}; filepath=nothing)
    nb_epochs = length(losses)
    p1 = plot(
        collect(1:nb_epochs),
        losses;
        title="Loss",
        xlabel="epochs",
        ylabel="loss",
        label=["train" "test"],
    )
    p2 = plot(
        collect(0:nb_epochs),
        gaps;
        title="Gap",
        xlabel="epochs",
        ylabel="ratio",
        label=["train" "test"],
    )
    pl = plot(p1, p2; layout=(1, 2))
    isnothing(filepath) || Plots.savefig(pl, filepath)
    return pl
end

function dijkstra_maximizer(θ::AbstractMatrix; kwargs...)
    # g = GridGraph(-θ; directions=QUEEN_DIRECTIONS)
    g = warcraft_grid_graph(-θ)
    # path = grid_dijkstra(g, 1, nv(g))
    p = dijkstra_shortest_paths(g, 1)
    path = get_path(p.parents, 1, nv(g))
    # y = get_path(path.parents, 1, nv(g))
    y = path_to_matrix(g, path)
    return y
end

"""
    grid_bellman_ford_warcraft(g, s, d, length_max)

Apply the Bellman-Ford algorithm on an `GridGraph` `g`, and return a `ShortestPathTree` with source `s` and destination `d`,
among the paths having length smaller than `length_max`.
"""
function grid_bellman_ford_warcraft(g, s::Integer, d::Integer, length_max::Int=nv(g))
    # Init storage
    parents = zeros(Int, nv(g), length_max + 1)
    dists = fill(Inf, nv(g), length_max + 1)
    # Add source
    dists[s, 1] = 0.0
    # Main loop
    for k in 1:length_max
        for v in vertices(g)
            for u in inneighbors(g, v)
                d_u = dists[u, k]
                if !isinf(d_u)
                    d_v = dists[v, k+1]
                    d_v_through_u = d_u + g.weights[u, v]  # GridGraphs.vertex_weight(g, v)
                    if isinf(d_v) || (d_v_through_u < d_v)
                        dists[v, k+1] = d_v_through_u
                        parents[v, k+1] = u
                    end
                end
            end
        end
    end
    # Get length of the shortest path
    k_short = argmin(dists[d, :])
    if isinf(dists[d, k_short])
        println("No shortest path with less than $length_max arcs")
        return Int[]
    end
    # Deduce the path
    v = d
    path = [v]
    k = k_short
    while v != s
        v = parents[v, k]
        if v == 0
            return Int[]
        else
            pushfirst!(path, v)
            k = k - 1
        end
    end
    return path
end

function bellman_maximizer(θ::AbstractMatrix; kwargs...)
    # g = GridGraph(-θ; directions=QUEEN_DIRECTIONS)
    g = warcraft_grid_graph(-θ)
    path = grid_bellman_ford_warcraft(g, 1, nv(g))
    # y = get_path(path, 1, nv(g))
    y = path_to_matrix(g, path)
    return y
end

"""
    average_tensor(x)

Average the tensor `x` along its third axis.
"""
function average_tensor(x)
    return sum(x; dims=[3]) / size(x)[3]
end

"""
    neg_tensor(x)

Compute minus softplus element-wise on tensor `x`.
"""
function neg_tensor(x)
    return -softplus.(x)
end

"""
    squeeze_last_dims(x)

Squeeze two last dimensions on tensor `x`.
"""
function squeeze_last_dims(x)
    return reshape(x, size(x, 1), size(x, 2))
end

struct WarcraftBenchmark end

function Utils.generate_dataset(
    ::WarcraftBenchmark,
    dataset_size::Int=10;
    type::Type=Float32,
)
    decompressed_path = datadep"warcraft/data"
    return create_dataset(decompressed_path, dataset_size)
end

function Utils.generate_maximizer(::WarcraftBenchmark; dijkstra=true)
    return dijkstra ? dijkstra_maximizer : bellman_maximizer
end

"""
    create_warcraft_embedding()

Create and return a `Flux.Chain` embedding for the Warcraft terrains, inspired by [differentiation of blackbox combinatorial solvers](https://github.com/martius-lab/blackbox-differentiation-combinatorial-solvers/blob/master/models.py).

The embedding is made as follows:
1) The first 5 layers of ResNet18 (convolution, batch normalization, relu, maxpooling and first resnet block).
2) An adaptive maxpooling layer to get a (12x12x64) tensor per input image.
3) An average over the third axis (of size 64) to get a (12x12x1) tensor per input image.
4) The element-wize `neg_tensor` function to get cell weights of proper sign to apply shortest path algorithms.
5) A squeeze function to forget the two last dimensions. 
"""
function Utils.generate_statistical_model(::WarcraftBenchmark)
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
