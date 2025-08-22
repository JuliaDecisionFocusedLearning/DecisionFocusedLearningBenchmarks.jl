"""
$TYPEDEF

Top k maximizer.
"""
struct TopKMaximizer
    k::Int
end

"""
$TYPEDSIGNATURES

Return the top k indices of `θ`.
"""
function (m::TopKMaximizer)(θ; kwargs...)
    N = length(θ)
    @assert N >= m.k "The length of θ must be at least k"
    solution = partialsortperm(θ, 1:(m.k); rev=true)
    res = falses(N)
    res[solution] .= 1
    return res
end
