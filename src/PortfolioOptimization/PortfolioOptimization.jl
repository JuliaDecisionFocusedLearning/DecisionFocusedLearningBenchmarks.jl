module PortfolioOptimization

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Distributions
using Flux: Chain, Dense
using Ipopt
using JuMP
using LinearAlgebra
using Random

"""
$TYPEDEF

Benchmark problem for the portfolio optimization problem.

Data is generated using the process described in: <https://arxiv.org/abs/2307.13565>.

# Fields
$TYPEDFIELDS
"""
struct PortfolioOptimizationBenchmark <: AbstractBenchmark
    "number of assets"
    d::Int
    "size of feature vectors"
    p::Int
    "hypermarameter for data generation"
    deg::Int
    "another hyperparameter, should be positive"
    ν::Float32
    "covariance matrix"
    Σ::Matrix{Float32}
    "maximum variance of portfolio"
    γ::Float32
    "useful for dataset generation"
    L::Matrix{Float32}
    "useful for dataset generation"
    f::Vector{Float32}
end

"""
$TYPEDSIGNATURES

Constructor for [`PortfolioOptimizationBenchmark`](@ref).
"""
function PortfolioOptimizationBenchmark(;
    d::Int=50, p::Int=5, deg::Int=1, ν::Float32=1.0f0, seed=0
)
    rng = MersenneTwister(seed)
    f = randn(rng, Float32, 4)
    L = Float32.(rand(rng, Uniform(-0.0025ν, 0.0025ν), d, 4))
    Σ = L * L' + (0.01f0ν)^2 * I
    e = ones(d) ./ d
    γ = 2.25e' * Σ * e
    return PortfolioOptimizationBenchmark(d, p, deg, ν, Σ, γ, L, f)
end

function Base.show(io::IO, bench::PortfolioOptimizationBenchmark)
    (; d, p, deg, ν) = bench
    return print(io, "PortfolioOptimizationBenchmark(d=$d, p=$p, deg=$deg, ν=$ν)")
end

"""
$TYPEDSIGNATURES

Create a function solving the MIQP formulation of the portfolio optimization problem.
"""
function Utils.generate_maximizer(bench::PortfolioOptimizationBenchmark)
    (; d, Σ, γ) = bench
    function portfolio_maximizer(θ)
        model = Model(Ipopt.Optimizer)
        set_silent(model)
        @variable(model, x[1:d] >= 0)

        @objective(model, Max, θ' * x)
        @constraint(model, sum(x) <= 1)
        @constraint(model, x' * Σ * x <= γ)

        optimize!(model)
        return value.(x)
    end
    return portfolio_maximizer
end

"""
$TYPEDSIGNATURES

Generate a dataset of labeled instances for the portfolio optimization problem.
"""
function Utils.generate_dataset(
    bench::PortfolioOptimizationBenchmark,
    dataset_size::Int=10;
    seed::Int=0,
    type::Type=Float32,
)
    (; d, p, deg, ν, L, f) = bench
    rng = MersenneTwister(seed)

    # Features
    features = [randn(rng, type, p) for _ in 1:dataset_size]

    # True weights
    B = rand(rng, Bernoulli(0.5), d, p)
    c̄ = [
        (0.05 / type(sqrt(p)) .* B * features[i] .+ 0.1^(1 / deg)) .^ deg for
        i in 1:dataset_size
    ]
    costs = [c̄ᵢ .+ L * f .+ 0.01 .* ν .* randn(rng, type, d) for c̄ᵢ in c̄]

    maximizer = Utils.generate_maximizer(bench)
    solutions = maximizer.(costs)

    return [
        DataSample(; x, θ_true, y_true) for
        (x, θ_true, y_true) in zip(features, costs, solutions)
    ]
end

"""
$TYPEDSIGNATURES

Initialize a linear model for `bench` using `Flux`.
"""
function Utils.generate_statistical_model(bench::PortfolioOptimizationBenchmark)
    (; p, d) = bench
    return Dense(p, d)
end

export PortfolioOptimizationBenchmark
export generate_dataset, generate_maximizer, generate_statistical_model

end
