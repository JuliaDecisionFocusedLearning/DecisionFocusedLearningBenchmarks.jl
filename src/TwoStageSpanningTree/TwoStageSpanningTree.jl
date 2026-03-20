module TwoStageSpanningTree

using DataStructures: IntDisjointSets, in_same_set, union!
using DocStringExtensions
using Flux
using Graphs
using GLPK
using HiGHS
using JuMP
using JuMP: MOI
using LinearAlgebra: dot
using Random
using Statistics: mean, quantile

using ..Utils

include("utils.jl")
include("instance.jl")
include("solution.jl")

include("algorithms/anticipative.jl")
include("algorithms/cut_generation.jl")
include("algorithms/benders_decomposition.jl")
include("algorithms/column_generation.jl")
include("algorithms/lagrangian_relaxation.jl")

include("learning/features.jl")

include("benchmark.jl")

export TwoStageSpanningTreeBenchmark
export TwoStageSpanningTreeInstance, nb_scenarios
export TwoStageSpanningTreeSolution,
    solution_value, is_feasible, solution_from_first_stage_forest
export kruskal, anticipative_solution
export cut_generation,
    benders_decomposition, column_generation, column_heuristic, lagrangian_relaxation

end
