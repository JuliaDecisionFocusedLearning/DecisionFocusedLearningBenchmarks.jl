"""
$TYPEDEF

Benchmark problem for the subset selection problem.
Reference: <https://arxiv.org/abs/2307.13565>.

# Fields
$TYPEDFIELDS
"""
struct SubsetSelectionBenchmark <: AbstractBenchmark
    "total number of items"
    n::Int
    "number of items to select"
    k::Int
end

function Base.show(io::IO, bench::SubsetSelectionBenchmark)
    (; n, k) = bench
    return print(io, "SubsetSelectionBenchmark(n=$n, k=$k)")
end

function SubsetSelectionBenchmark(; n::Int=25, k::Int=5)
    @assert n >= k
    return SubsetSelectionBenchmark(n, k)
end

function top_k(v::AbstractVector, k::Int)
    indices = sortperm(v; rev=true)[1:k]
    res = falses(length(v))
    res[indices] .= true
    return res
end

"""
$TYPEDSIGNATURES

Return a top k maximizer.
"""
function Utils.generate_maximizer(bench::SubsetSelectionBenchmark)
    (; k) = bench
    return x -> top_k(x, k)
end

"""
$TYPEDSIGNATURES

Generate a dataset of labeled instances for the subset selection problem.
"""
function Utils.generate_dataset(
    bench::SubsetSelectionBenchmark, dataset_size::Int=10; seed::Int=0
)
    (; n, k) = bench
    rng = MersenneTwister(seed)
    features = [randn(rng, n) for _ in 1:dataset_size]
    solutions = top_k.(features, k)
    return InferOptDataset(; features, solutions)
end

"""
$TYPEDSIGNATURES

Initialize a linear model for `bench` using `Flux`.
"""
function Utils.generate_statistical_model(bench::SubsetSelectionBenchmark)
    (; n) = bench
    return Chain(Dense(n, n))
end
