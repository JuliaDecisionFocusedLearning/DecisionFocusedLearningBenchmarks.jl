module DynamicAssortment

using ..Utils

using CommonRLInterface: CommonRLInterface, AbstractEnv
using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using Distributions: Uniform, Categorical
using LinearAlgebra: dot
using Random: Random, AbstractRNG, MersenneTwister
using Statistics: mean

using Flux: Chain, Dense
# using Flux.Optimise
# using InferOpt
# using Random
# using JLD2
# using Plots
# using Distributions
# using LinearAlgebra
using Combinatorics: combinations

include("environment.jl")

struct DynamicAssortmentBenchmark <: AbstractDynamicBenchmark end

function Utils.generate_sample(::DynamicAssortmentBenchmark)
    return DataSample(; instance=Instance())
end

function Utils.generate_maximizer(::DynamicAssortmentBenchmark)
    return DAP_optimization
end

export DynamicAssortmentBenchmark

end
