"""
$TYPEDEF

Data sample data structure.

# Fields
$TYPEDFIELDS
"""
@kwdef struct DataSample{
    I,F<:AbstractArray,S<:Union{AbstractArray,Nothing},C<:Union{AbstractArray,Nothing}
}
    "features"
    x::F
    "target cost parameters (optional)"
    θ_true::C = nothing
    "target solution (optional)"
    y_true::S = nothing
    "instance object (optional)"
    instance::I = nothing
end

function _transform(t, sample::DataSample; kwargs...)
    (; instance, x, θ_true, y_true) = sample
    return DataSample(; instance, x=StatsBase.transform(t, x; kwargs...), θ_true, y_true)
end

function _reconstruct(t, sample::DataSample; kwargs...)
    (; instance, x, θ_true, y_true) = sample
    return DataSample(; instance, x=StatsBase.reconstruct(t, x; kwargs...), θ_true, y_true)
end

"""
$TYPEDSIGNATURES

Compute the mean and standard deviation of the features in the dataset.
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

# TODO: reconstruct, transform!, reconstruct!

function StatsBase.reconstruct(t, dataset::AbstractVector{<:DataSample})
    return map(dataset) do d
        (; instance, x, θ_true, y_true) = d
        DataSample(; instance, x=StatsBase.reconstruct(t, x), θ_true, y_true)
    end
end
