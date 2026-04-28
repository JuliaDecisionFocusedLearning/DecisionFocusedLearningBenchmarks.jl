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
    rows, cols = bench.grid_size
    # Show only the known graph structure (no edge costs)
    interior_xs = [
        c for r in 1:rows for
        c in 1:cols if !(r == 1 && c == 1) && !(r == rows && c == cols)
    ]
    interior_ys = [
        r for r in 1:rows for
        c in 1:cols if !(r == 1 && c == 1) && !(r == rows && c == cols)
    ]
    pl = Plots.plot(;
        xlim=(0.5, cols + 0.5),
        ylim=(0.5, rows + 0.5),
        yflip=true,
        aspect_ratio=:equal,
        legend=:topright,
        title="Grid graph ($(rows)×$(cols))",
        framestyle=:box,
        grid=false,
        kwargs...,
    )
    Plots.scatter!(
        pl,
        interior_xs,
        interior_ys;
        color=:lightgray,
        markersize=8,
        markerstrokecolor=:gray,
        markerstrokewidth=1,
        label=false,
    )
    Plots.scatter!(
        pl,
        [1],
        [1];
        color=:seagreen,
        markersize=10,
        markershape=:square,
        label="source",
        markerstrokewidth=0,
    )
    Plots.scatter!(
        pl,
        [cols],
        [rows];
        color=:crimson,
        markersize=10,
        markershape=:square,
        label="sink",
        markerstrokewidth=0,
    )
    return pl
end

function plot_solution(bench::FixedSizeShortestPathBenchmark, sample::DataSample; kwargs...)
    x = sample.x
    p_feat = length(x)
    weight_grid, path_grid = _grid_matrices(bench, sample.θ, sample.y)
    rows, cols = bench.grid_size

    p_x = Plots.bar(
        1:p_feat,
        Float64.(x);
        legend=false,
        xlabel="Feature",
        ylabel="Value",
        title="x (features, observable)",
        color=:steelblue,
        xticks=1:p_feat,
    )
    p1 = Plots.heatmap(
        weight_grid;
        yflip=true,
        aspect_ratio=:equal,
        title="Edge weights θ",
        colorbar=true,
        framestyle=:none,
    )
    p2 = Plots.heatmap(
        weight_grid;
        yflip=true,
        aspect_ratio=:equal,
        title="Shortest path y",
        colorbar=false,
        framestyle=:none,
        color=:Blues,
    )
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

    l = Plots.@layout [a{0.25h}; [b c]]
    return Plots.plot(p_x, p1, p2; layout=l, size=(700, 500), kwargs...)
end
