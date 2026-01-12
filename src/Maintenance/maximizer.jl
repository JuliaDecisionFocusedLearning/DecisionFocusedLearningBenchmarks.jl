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

    positive_indices = findall(x -> x > 0, θ)
    nb_positive = length(positive_indices)
    res = falses(N)

    if nb_positive == 0
        return res
    elseif nb_positive <= m.k
        res[positive_indices] .= true
        return res
    else
        idx = partialsortperm(θ[positive_indices], 1:(m.k); rev=true)
        res[positive_indices[idx]] .= true
        return res
    end
end
