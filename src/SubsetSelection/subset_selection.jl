"""
$TYPEDEF

Benchmark problem for the subset selection problem.
Reference: <https://arxiv.org/abs/2307.13565>.

The goal is to select the best `k` items from a set of `n` items,
without knowing their values, but only observing some features.

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
    @assert n >= k "number of items n must be greater than k"
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
    return Base.Fix2(top_k, k)
end

"""
$TYPEDSIGNATURES

Generate a dataset of labeled instances for the subset selection problem.
The mapping between features and cost is identity.
"""
function Utils.generate_dataset(
    bench::SubsetSelectionBenchmark, dataset_size::Int=10; seed::Int=0
)
    (; n, k) = bench
    rng = MersenneTwister(seed)
    features = [randn(rng, n) for _ in 1:dataset_size]
    costs = copy(features)  # we assume that the cost is the same as the feature
    solutions = top_k.(features, k)
    return InferOptDataset(; features, solutions, costs)
end

"""
$TYPEDSIGNATURES

Initialize a linear model for `bench` using `Flux`.
"""
function Utils.generate_statistical_model(bench::SubsetSelectionBenchmark)
    (; n) = bench
    return Chain(Dense(n => n; bias=false))
end
