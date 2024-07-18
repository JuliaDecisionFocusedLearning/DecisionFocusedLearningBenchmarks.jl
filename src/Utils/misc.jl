"""
    neg_tensor(x)

Compute minus softplus element-wise on tensor `x`.
"""
function neg_tensor(x)
    return -softplus.(x)
end

"""
    squeeze_last_dims(x)

Squeeze two last dimensions on tensor `x`.
"""
function squeeze_last_dims(x)
    return reshape(x, size(x, 1), size(x, 2))
end

"""
    average_tensor(x)

Average the tensor `x` along its third axis.
"""
function average_tensor(x)
    return sum(x; dims=[3]) / size(x)[3]
end
