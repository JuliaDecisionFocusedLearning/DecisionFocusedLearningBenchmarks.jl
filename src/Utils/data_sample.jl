"""
$TYPEDEF

Data sample data structure.

# Fields
$TYPEDFIELDS
"""
@kwdef struct DataSample{F,S,C,I}
    "features"
    x::F
    "costs"
    θ::C = nothing
    "solution"
    y::S = nothing
    "instance"
    instance::I = nothing
end
