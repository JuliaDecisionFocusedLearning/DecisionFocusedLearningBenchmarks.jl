"""
$TYPEDSIGNATURES

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
$TYPEDSIGNATURES

Create the dataset corresponding to the data located at `decompressed_path`, possibly sub-sampling `nb_samples` points.
The dataset is made of images of Warcraft terrains, cell cost labels and shortest path labels.
It is a `Vector` of tuples, each `Tuple` being a dataset point.
"""
function create_dataset(decompressed_path::String, nb_samples::Int)
    N = min(nb_samples, 10000)
    terrain_images, terrain_labels, terrain_weights = read_dataset(
        decompressed_path, "train"
    )
    X = [
        reshape(terrain_images[:, :, :, i], (size(terrain_images[:, :, :, i])..., 1)) for
        i in 1:N
    ]
    Y = [BitMatrix(terrain_labels[:, :, i]) for i in 1:N]
    WG = [-terrain_weights[:, :, i] for i in 1:N]
    return [DataSample(; x, y=y_true, θ=θ_true) for (x, y_true, θ_true) in zip(X, Y, WG)]
end

"""
$TYPEDSIGNATURES

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
                    d_v = dists[v, k + 1]
                    d_v_through_u = d_u + g.weights[u, v]  # GridGraphs.vertex_weight(g, v)
                    if isinf(d_v) || (d_v_through_u < d_v)
                        dists[v, k + 1] = d_v_through_u
                        parents[v, k + 1] = u
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

"""
$TYPEDSIGNATURES

Computes the longest path in given grid graph weights by computing the shortest path in the graph with opposite weights.
Using the Ford-Bellman dynamic programming algorithm.
"""
function bellman_maximizer(θ::AbstractMatrix; kwargs...)
    g = grid_graph(-θ)
    path = grid_bellman_ford_warcraft(g, 1, nv(g))
    y = path_to_matrix(path, 12, 12)
    return y
end

"""
$TYPEDSIGNATURES

Computes the longest path in given grid graph weights by computing the shortest path in the graph with opposite weights.
Using the Dijkstra algorithm.

!!! warning
    Only works on graph with positive weights, i.e. if `θ` only contains negative values.
"""
function dijkstra_maximizer(θ::AbstractMatrix; kwargs...)
    g = grid_graph(-θ)
    p = dijkstra_shortest_paths(g, 1)
    path = get_path(p.parents, 1, nv(g))
    y = path_to_matrix(path, 12, 12)
    return y
end
