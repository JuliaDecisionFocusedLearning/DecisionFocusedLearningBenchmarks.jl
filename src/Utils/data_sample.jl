"""
$TYPEDEF

Data sample data structure.

# Fields
$TYPEDFIELDS
"""
@kwdef struct DataSample{
    I,
    F<:Union{AbstractArray,Nothing},
    S<:Union{AbstractArray,Nothing},
    C<:Union{AbstractArray,Nothing},
}
    "features"
    x::F = nothing
    "target cost parameters (optional)"
    θ_true::C = nothing
    "target solution (optional)"
    y_true::S = nothing
    "instance object (optional)"
    instance::I = nothing
end

"""
$TYPEDSIGNATURES

Fit the given transform type (`ZScoreTransform` or `UnitRangeTransform`) on the dataset.
"""
function StatsBase.fit(transform_type, dataset::AbstractVector{<:DataSample}; kwargs...)
    x = hcat([d.x for d in dataset]...)
    return StatsBase.fit(transform_type, x; kwargs...)
end

"""
$TYPEDSIGNATURES

Transform the features in the dataset.
"""
function StatsBase.transform(t, dataset::AbstractVector{<:DataSample})
    return map(dataset) do d
        (; instance, x, θ_true, y_true) = d
        DataSample(; instance, x=StatsBase.transform(t, x), θ_true, y_true)
    end
end

"""
$TYPEDSIGNATURES

Transform the features in the dataset in place.
"""
function StatsBase.transform!(t, dataset::AbstractVector{<:DataSample})
    for d in dataset
        StatsBase.transform!(t, d.x)
    end
end

"""
$TYPEDSIGNATURES

Reconstruct the features in the dataset.
"""
function StatsBase.reconstruct(t, dataset::AbstractVector{<:DataSample})
    return map(dataset) do d
        (; instance, x, θ_true, y_true) = d
        DataSample(; instance, x=StatsBase.reconstruct(t, x), θ_true, y_true)
    end
end

"""
$TYPEDSIGNATURES

Reconstruct the features in the dataset in place.
"""
function StatsBase.reconstruct!(t, dataset::AbstractVector{<:DataSample})
    for d in dataset
        StatsBase.reconstruct!(t, d.x)
    end
end
