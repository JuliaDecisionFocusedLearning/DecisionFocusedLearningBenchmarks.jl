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
