"""
$TYPEDEF

Dataset data structure.

# Fields
$TYPEDFIELDS

`Base.length` and `Base.getindex` are implemented for this type.
"""
@kwdef struct InferOptDataset{F,I,C,S}
    "the only mandatory field, vector of features"
    features::Vector{F}
    "list of instances, can be set to `nothing` if not needed"
    instances::I = nothing
    "groundtruth costs of each instance, set to `nothing` if not available"
    costs::C = nothing
    "groundtruth solutions of each instance, set to `nothing` if not available"
    solutions::S = nothing
end

Base.length(dataset::InferOptDataset) = length(dataset.features)

my_getindex(v::AbstractVector, idx...) = getindex(v, idx...)
my_getindex(::Nothing, idx...) = nothing

function Base.getindex(dataset::InferOptDataset, idx...)
    features = getindex(dataset.features, idx...)
    instances = my_getindex(dataset.instances, idx...)
    costs = my_getindex(dataset.costs, idx...)
    solutions = my_getindex(dataset.solutions, idx...)
    return InferOptDataset(; features, instances, costs, solutions)
end

function Base.getindex(dataset::InferOptDataset, idx::Int)
    features = getindex(dataset.features, idx)
    instance = my_getindex(dataset.instances, idx)
    costs = my_getindex(dataset.costs, idx)
    solution = my_getindex(dataset.solutions, idx)
    return (; features, instance, costs, solution)
end
