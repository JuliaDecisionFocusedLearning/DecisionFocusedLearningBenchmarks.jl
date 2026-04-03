import Graphs: edges, src, dst

has_visualization(::FixedSizeShortestPathBenchmark) = true

"""
Map edge weights to a (rows × cols) vertex weight matrix by averaging incident edge weights,
and return a boolean (rows × cols) matrix marking vertices on the path.
"""
function _grid_matrices(bench::FixedSizeShortestPathBenchmark, θ, y)
    rows, cols = bench.grid_size
    n_v = rows * cols
    g = bench.graph

    # Vertex weights: mean of absolute weights of incident edges
    v_weights = zeros(Float64, n_v)
    v_counts = zeros(Int, n_v)
    for (i, e) in enumerate(edges(g))
        v_weights[src(e)] += abs(θ[i])
        v_counts[src(e)] += 1
        v_weights[dst(e)] += abs(θ[i])
        v_counts[dst(e)] += 1
    end
    v_weights ./= max.(v_counts, 1)

    # Path vertices
    on_path = falses(n_v)
    for (i, e) in enumerate(edges(g))
        if y[i]
            on_path[src(e)] = true
            on_path[dst(e)] = true
        end
    end

    # Reshape to (rows, cols): vertex v → row ceil(v/cols), col ((v-1)%cols)+1
    weight_grid = reshape(v_weights, cols, rows)'
    path_grid = reshape(on_path, cols, rows)'
    return weight_grid, path_grid
end

function plot_instance(bench::FixedSizeShortestPathBenchmark, sample::DataSample; kwargs...)
    weight_grid, _ = _grid_matrices(bench, sample.θ, falses(length(sample.θ)))
    return Plots.heatmap(
        weight_grid;
        yflip=true,
        aspect_ratio=:equal,
        title="Edge weights (per vertex)",
        colorbar=true,
        kwargs...,
    )
end

function plot_solution(bench::FixedSizeShortestPathBenchmark, sample::DataSample; kwargs...)
    weight_grid, path_grid = _grid_matrices(bench, sample.θ, sample.y)
    rows, cols = bench.grid_size

    p1 = Plots.heatmap(
        weight_grid;
        yflip=true,
        aspect_ratio=:equal,
        title="Edge weights",
        colorbar=true,
        framestyle=:none,
    )

    p2 = Plots.heatmap(
        weight_grid;
        yflip=true,
        aspect_ratio=:equal,
        title="Shortest path",
        colorbar=false,
        framestyle=:none,
        color=:Blues,
    )
    # Highlight path vertices with scatter
    path_xs = Int[]
    path_ys = Int[]
    for r in 1:rows, c in 1:cols
        if path_grid[r, c]
            push!(path_xs, c)
            push!(path_ys, r)
        end
    end
    Plots.scatter!(
        p2, path_xs, path_ys; color=:white, markersize=6, markerstrokewidth=0, label=false
    )

    return Plots.plot(p1, p2; layout=(1, 2), size=(700, 320), kwargs...)
end
