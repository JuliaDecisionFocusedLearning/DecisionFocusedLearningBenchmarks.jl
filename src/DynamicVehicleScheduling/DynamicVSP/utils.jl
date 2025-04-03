"""
$TYPEDEF

Basic point structure.
"""
struct Point{T}
    x::T
    y::T
end

Base.show(io::IO, p::Point) = print(io, "($(p.x), $(p.y))")
