struct ShortestPathProblem end

"""
$TYPEDEF

Benchmark dataset using a shortest path CO layer.

# Fields
$TYPEDFIELDS
"""
struct ShortestPathBenchmark{F<:AbstractVector,C<:AbstractVector,P<:AbstractVector,M} <:
       AbstractBenchmark
    "grid size of graphs"
    grid_size::Tuple{Int,Int}
    "vector of feature arrays"
    features::F
    "label 'true' optimization parameters"
    optimization_parameters::C
    "label solutions"
    solutions::P
    "(combinatorial) optimization (arg) maximizer"
    maximizer::M
end

"""
$TYPEDSIGNATURES

Custom constructor for [`ShortestPathBenchmark`](@ref).
"""
function ShortestPathBenchmark(
    dataset_size=10; p=5, grid_size=(5, 5), deg=1, ν=0.5, seed=0, type=Float32
)
    # Set seed
    rng = MersenneTwister(seed)

    # Compute directed grid graph
    g = DiGraph(collect(edges(Graphs.grid(grid_size))))
    E = Graphs.ne(g)
    V = Graphs.nv(g)

    # Features
    features = [randn(rng, type, p) for _ in 1:dataset_size]

    # True weights
    B = rand(rng, Bernoulli(0.5), E, p)
    ξ = [rand(rng, Uniform(1 - ν, 1 + ν), E) for _ in 1:dataset_size]
    costs = [(1 .+ (3 .+ B * zᵢ ./ sqrt(p)) .^ deg) .* ξᵢ for (ξᵢ, zᵢ) in zip(ξ, features)]
    I = [src(e) for e in edges(g)]
    J = [dst(e) for e in edges(g)]

    # Maximizer
    function shortest_path_maximizer(θ; kwargs...)
        weights = sparse(I, J, -θ, V, V)
        parents = Graphs.bellman_ford_shortest_paths(g, 1, weights).parents
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

    # Label solutions
    solutions = shortest_path_maximizer.(.-costs)
    return ShortestPathBenchmark(
        grid_size, features, costs, solutions, shortest_path_maximizer
    )
end

function objective_value(θ, y)
    return dot(θ, y)
end

function compute_gap(bench::ShortestPathBenchmark, model)
    res = 0.0
    for (x, ȳ, θ̄) in zip(bench.features, bench.solutions, bench.optimization_parameters)
        θ = model(x)
        y = bench.maximizer(θ)
        val = objective_value(θ̄, ȳ)
        res += (objective_value(θ̄, y) - val) / val
    end
    return res / length(bench.features)
end
