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

function _plot_grid(
    bench::FixedSizeShortestPathBenchmark;
    grid=nothing,
    title="",
    colorbar=false,
    color=:viridis,
    path_grid=nothing,
    kwargs...,
)
    rows, cols = bench.grid_size
    if isnothing(grid)
        grid = ones(rows, cols)
    end
    pl = Plots.heatmap(
        grid;
        yflip=true,
        aspect_ratio=:equal,
        title=title,
        colorbar=colorbar,
        framestyle=:none,
        color=color,
        kwargs...,
    )
    Plots.vline!(pl, (0.5):1:(cols + 0.5); color=:gray, lw=0.5, label=false)
    Plots.hline!(pl, (0.5):1:(rows + 0.5); color=:gray, lw=0.5, label=false)
    if !isnothing(path_grid)
        path_xs = Int[]
        path_ys = Int[]
        for r in 1:rows, c in 1:cols
            if path_grid[r, c]
                push!(path_xs, c)
                push!(path_ys, r)
            end
        end
        Plots.scatter!(
            pl, path_xs, path_ys; color=:white, markersize=6, markerstrokewidth=0, label=false
        )
    end
    Plots.scatter!(
        pl, [1], [1]; color=:seagreen, markersize=8, markershape=:square, label=false
    )
    Plots.scatter!(
        pl,
        [cols],
        [rows];
        color=:crimson,
        markersize=8,
        markershape=:square,
        label=false,
    )
    return pl
end

function plot_context(bench::FixedSizeShortestPathBenchmark, sample::DataSample; kwargs...)
    x = sample.x
    p_feat = length(x)
    rows, cols = bench.grid_size

    p_x = Plots.bar(
        1:p_feat,
        Float64.(x);
        legend=false,
        xlabel="Feature",
        ylabel="Value",
        title="x (features)",
        color=:steelblue,
        xticks=1:p_feat,
    )
    p_grid = _plot_grid(bench; title="Grid graph ($(rows)×$(cols))", color=:grays)

    l = Plots.@layout [a{0.35w} b]
    return Plots.plot(p_x, p_grid; layout=l, size=(700, 300), kwargs...)
end

function plot_sample(bench::FixedSizeShortestPathBenchmark, sample::DataSample; kwargs...)
    x = sample.x
    p_feat = length(x)
    weight_grid, path_grid = _grid_matrices(bench, sample.θ, sample.y)

    p_x = Plots.bar(
        1:p_feat,
        Float64.(x);
        legend=false,
        xlabel="Feature",
        ylabel="Value",
        title="x (features)",
        color=:steelblue,
        xticks=1:p_feat,
    )
    p1 = _plot_grid(bench; grid=weight_grid, title="Edge weights θ", colorbar=true)
    p2 = _plot_grid(
        bench; grid=weight_grid, title="Shortest path y", color=:Blues, path_grid=path_grid
    )

    l = Plots.@layout [a{0.25h}; [b c]]
    return Plots.plot(p_x, p1, p2; layout=l, size=(700, 500), kwargs...)
end
