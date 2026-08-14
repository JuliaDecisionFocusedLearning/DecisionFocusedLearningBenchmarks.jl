"""
$TYPEDSIGNATURES

Return a `StableRNG` seeded with `seed`, or with a random seed if `seed` is `nothing`.
"""
make_rng(seed::Integer) = StableRNG(seed)
make_rng(::Nothing) = StableRNG(rand(RandomDevice(), UInt))

"""
$TYPEDEF

Simple callable wrapper around a weight matrix: `x ↦ weights * x`.

# Fields
$TYPEDFIELDS
"""
struct LinearModel{W<:AbstractMatrix}
    "weight matrix"
    weights::W
end

(m::LinearModel)(x) = m.weights * x

"""
$TYPEDSIGNATURES

Compute minus softplus element-wise on tensor `x`.
"""
function neg_tensor(x)
    return -softplus.(x)
end

"""
$TYPEDSIGNATURES

Squeeze two last dimensions on tensor `x`.
"""
function squeeze_last_dims(x)
    return reshape(x, size(x, 1), size(x, 2))
end

"""
$TYPEDSIGNATURES

Average the tensor `x` along its third axis.
"""
function average_tensor(x)
    return sum(x; dims=[3]) / size(x)[3]
end
