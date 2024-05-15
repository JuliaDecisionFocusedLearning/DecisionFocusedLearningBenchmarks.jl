module PortfolioOptimization

using ..Utils
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Distributions
using Flux: Chain, Dense
using Ipopt
using JuMP
using LinearAlgebra
using Random

include("portfolio_optimization.jl")

export PortfolioOptimizationBenchmark
export generate_dataset, generate_statistical_model, generate_maximizer

end
