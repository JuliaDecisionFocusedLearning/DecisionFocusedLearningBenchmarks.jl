"""
$TYPEDEF

Benchmark problem for the shortest path problem.
In this benchmark, all graphs are acyclic directed grids, all of the same size `grid_size`.
Features are given at instance level (one dimensional vector of length `p` for each graph).

Data is generated using the process described in: <https://arxiv.org/abs/2307.13565>.

# Fields
$TYPEDFIELDS
"""
struct ShortestPathBenchmark <: AbstractBenchmark
    "grid graph instance"
    graph::SimpleDiGraph{Int64}
    "grid size of graphs"
    grid_size::Tuple{Int,Int}
    "size of feature vectors"
    p::Int
    "degree of formula between features and true weights"
    deg::Int
    "multiplicative noise for true weights sampled between [1-ν, 1+ν]"
    ν::Float64
end

function Base.show(io::IO, bench::ShortestPathBenchmark)
    (; grid_size, p, deg, ν) = bench
    return print(io, "ShortestPathBenchmark(grid_size=$grid_size, p=$p, deg=$deg, ν=$ν)")
end

"""
$TYPEDSIGNATURES

Constructor for [`ShortestPathBenchmark`](@ref).
"""
function ShortestPathBenchmark(;
    grid_size::Tuple{Int,Int}=(5, 5), p::Int=5, deg::Int=1, ν=0.0
)
    @assert ν >= 0.0 && ν <= 1.0
    g = DiGraph(collect(edges(Graphs.grid(grid_size))))
    return ShortestPathBenchmark(g, grid_size, p, deg, ν)
end

"""
$TYPEDSIGNATURES

Outputs a function that computes the longest path on the grid graph, given edge weights θ as input.

```julia
maximizer = generate_maximizer(bench)
maximizer(θ)
```
"""
function generate_maximizer(bench::ShortestPathBenchmark; use_dijkstra=true)
    g = bench.graph
    V = Graphs.nv(g)
    E = Graphs.ne(g)

    I = [src(e) for e in edges(g)]
    J = [dst(e) for e in edges(g)]
    algo =
        use_dijkstra ? Graphs.dijkstra_shortest_paths : Graphs.bellman_ford_shortest_paths

    function shortest_path_maximizer(θ; kwargs...)
        weights = sparse(I, J, -θ, V, V)
        parents = algo(g, 1, weights).parents
        y = falses(V, V)
        u = V
        while u != 1
            prev = parents[u]
            y[prev, u] = true
            u = prev
        end

        solution = falses(E)
        for (i, edge) in enumerate(edges(g))
            if y[src(edge), dst(edge)]
                solution[i] = true
            end
        end
        return solution
    end

    return shortest_path_maximizer
end

"""
$TYPEDSIGNATURES

Constructor for [`ShortestPathBenchmark`](@ref).
"""
function generate_dataset(
    bench::ShortestPathBenchmark, dataset_size::Int=10; seed::Int=0, type::Type=Float32
)
    # Set seed
    rng = MersenneTwister(seed)
    (; graph, p, deg, ν) = bench

    E = Graphs.ne(graph)

    # Features
    features = [randn(rng, type, p) for _ in 1:dataset_size]

    # True weights
    B = rand(rng, Bernoulli(0.5), E, p)
    ξ = if ν == 0.0
        [ones(type, E) for _ in 1:dataset_size]
    else
        [rand(rng, Uniform{type}(1 - ν, 1 + ν), E) for _ in 1:dataset_size]
    end
    costs = [
        (1 .+ (3 .+ B * zᵢ ./ type(sqrt(p))) .^ deg) .* ξᵢ for (ξᵢ, zᵢ) in zip(ξ, features)
    ]

    shortest_path_maximizer = generate_maximizer(bench)

    # Label solutions
    solutions = shortest_path_maximizer.(.-costs)
    return (; features, costs, solutions)
end

"""
$TYPEDSIGNATURES

Initialize a linear model for `bench` using `Flux`.
"""
function generate_statistical_model(bench::ShortestPathBenchmark)
    (; p, graph) = bench
    return Chain(Dense(p, ne(graph)))
end

function objective_value(::ShortestPathBenchmark, θ, y)
    return dot(θ, y)
end

function compute_gap(
    bench::ShortestPathBenchmark, model, features, costs, solutions, maximizer
)
    res = 0.0
    for (x, ȳ, θ̄) in zip(features, solutions, costs)
        θ = model(x)
        y = maximizer(θ)
        val = objective_value(bench, θ̄, ȳ)
        res += (objective_value(bench, θ̄, y) - val) / val
    end
    return res / length(features)
end
