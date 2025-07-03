module FixedSizeShortestPath

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Distributions
using Flux: Chain, Dense
using Graphs
using LinearAlgebra
using Random
using SparseArrays

"""
$TYPEDEF

Benchmark problem for the shortest path problem.
In this benchmark, all graphs are acyclic directed grids, all of the same size `grid_size`.
Features are given at instance level (one dimensional vector of length `p` for each graph).

Data is generated using the process described in: <https://arxiv.org/abs/2307.13565>.

# Fields
$TYPEDFIELDS
"""
struct FixedSizeShortestPathBenchmark <: AbstractBenchmark
    "grid graph instance"
    graph::SimpleDiGraph{Int64}
    "grid size of graphs"
    grid_size::Tuple{Int,Int}
    "size of feature vectors"
    p::Int
    "degree of formula between features and true weights"
    deg::Int
    "multiplicative noise for true weights sampled between [1-ν, 1+ν], should be between 0 and 1"
    ν::Float32
end

function Base.show(io::IO, bench::FixedSizeShortestPathBenchmark)
    (; grid_size, p, deg, ν) = bench
    return print(
        io, "FixedSizeShortestPathBenchmark(grid_size=$grid_size, p=$p, deg=$deg, ν=$ν)"
    )
end

"""
$TYPEDSIGNATURES

Constructor for [`FixedSizeShortestPathBenchmark`](@ref).
"""
function FixedSizeShortestPathBenchmark(;
    grid_size::Tuple{Int,Int}=(5, 5), p::Int=5, deg::Int=1, ν=0.0f0
)
    @assert ν >= 0.0 && ν <= 1.0
    g = DiGraph(collect(edges(Graphs.grid(grid_size))))
    return FixedSizeShortestPathBenchmark(g, grid_size, p, deg, ν)
end

function Utils.objective_value(
    ::FixedSizeShortestPathBenchmark, θ::AbstractArray, y::AbstractArray
)
    return -dot(θ, y)
end

"""
$TYPEDSIGNATURES

Outputs a function that computes the longest path on the grid graph, given edge weights θ as input.

```julia
maximizer = generate_maximizer(bench)
maximizer(θ)
```
"""
function Utils.generate_maximizer(bench::FixedSizeShortestPathBenchmark; use_dijkstra=true)
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

function Utils.generate_sample(
    bench::FixedSizeShortestPathBenchmark, rng::AbstractRNG; type::Type=Float32
)
    (; graph, p, deg, ν) = bench
    features = randn(rng, Float32, bench.p)
    E = Graphs.ne(graph)
    # True weights
    B = rand(rng, Bernoulli(0.5), E, p)
    ξ = if ν == 0.0
        ones(type, E)
    else
        rand(rng, Uniform{type}(1 - ν, 1 + ν), E)
    end
    costs = -(1 .+ (3 .+ B * features ./ type(sqrt(p))) .^ deg) .* ξ

    maximizer = Utils.generate_maximizer(bench)
    solution = maximizer(costs)
    return DataSample(; x=features, θ_true=costs, y_true=solution)
end

"""
$TYPEDSIGNATURES

Initialize a linear model for `bench` using `Flux`.
"""
function Utils.generate_statistical_model(bench::FixedSizeShortestPathBenchmark)
    (; p, graph) = bench
    return Chain(Dense(p, ne(graph)))
end

export FixedSizeShortestPathBenchmark
export generate_dataset, generate_maximizer, generate_statistical_model

end
