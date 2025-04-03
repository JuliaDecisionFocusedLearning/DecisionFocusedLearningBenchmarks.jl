abstract type AbstractDynamicPolicy end

function (π::AbstractDynamicPolicy)(env; kwargs...)
    throw("Not implemented")
end
