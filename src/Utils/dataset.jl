"""
$TYPEDEF

Dataset data structure.

# Fields
$TYPEDFIELDS
"""
struct InferOptDataset{I,F,C,S}
    "the only mandatory field, vector of features"
    features::F
    "list of instances, can be set to `nothing` if not needed"
    instances::I
    "groundtruth costs of each instance, if available"
    costs::C
    "groundtruth solutions of each instance, if available"
    solutions::S
end

"""
$TYPEDSIGNATURES
"""
Base.length(dataset::InferOptDataset) = length(dataset.features)
