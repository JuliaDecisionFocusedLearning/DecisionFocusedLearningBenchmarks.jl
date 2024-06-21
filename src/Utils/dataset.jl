"""
$TYPEDEF

Dataset data structure.

# Fields
$TYPEDFIELDS
"""
struct InferOptDataset{F,I,C,S}
    "the only mandatory field, vector of features"
    features::Vector{F}
    "list of instances, can be set to `nothing` if not needed"
    instances::I
    "groundtruth costs of each instance, set to `nothing` if not available"
    costs::C
    "groundtruth solutions of each instance, set to `nothing` if not available"
    solutions::S
end

"""
$TYPEDSIGNATURES
"""
Base.length(dataset::InferOptDataset) = length(dataset.features)
