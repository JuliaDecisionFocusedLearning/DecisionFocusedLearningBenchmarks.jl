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
    "input features (optional)"
    x::F = nothing
    "intermediate cost parameters (optional)"
    θ::C = nothing
    "output solution (optional)"
    y::S = nothing
    "additional information, usually the instance (optional)"
    info::I = nothing
end

function Base.show(io::IO, d::DataSample)
    fields = String[]
    if !isnothing(d.x)
        push!(fields, "x=$(d.x)")
    end
    if !isnothing(d.θ)
        push!(fields, "θ_true=$(d.θ)")
    end
    if !isnothing(d.y)
        push!(fields, "y_true=$(d.y)")
    end
    if !isnothing(d.info)
        push!(fields, "instance=$(d.info)")
    end
    return print(io, "DataSample(", join(fields, ", "), ")")
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
        (; info, x, θ, y) = d
        DataSample(; info, x=StatsBase.transform(t, x), θ, y)
    end
end

"""
$TYPEDSIGNATURES

Transform the features in the dataset, in place.
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
        (; info, x, θ, y) = d
        DataSample(; info, x=StatsBase.reconstruct(t, x), θ, y)
    end
end

"""
$TYPEDSIGNATURES

Reconstruct the features in the dataset, in place.
"""
function StatsBase.reconstruct!(t, dataset::AbstractVector{<:DataSample})
    for d in dataset
        StatsBase.reconstruct!(t, d.x)
    end
end
