"""
$TYPEDEF

Top k maximizer.
"""
struct TopKPositiveMaximizer
    k::Int
end

"""
$TYPEDSIGNATURES

Return the top k indices of `θ`.
"""

function (m::TopKPositiveMaximizer)(θ; kwargs...)
    N = length(θ)
    
    sorted_indices = sortperm(θ; rev=true)
    positive_indices = filter(i -> θ[i] > 0, sorted_indices)
    solution = positive_indices[1:min(m.k, length(positive_indices))]
    
    res = falses(N)
    res[solution] .= 1
    return res
end