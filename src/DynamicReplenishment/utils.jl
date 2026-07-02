function compute_μ_σ_matrix(X::Matrix{Float64})
    μ = mean(X; dims=1)
    σ = std(X; dims=1)
    for i in eachindex(σ)
        if abs(σ[i]) < 1e-6
            σ[i] = 1.0
        end
    end
    return vec(μ), vec(σ)
end

"""
    reduce_data!(X, σ)

Reduce X with σ, without centering it.
"""
function reduce_data!(X::Matrix{Float64}, σ; center=false, μ=nothing)
    if center
        @assert μ !== nothing "μ must be provided if center=true"
        for features in eachrow(X)
            @. features = (features - μ) / σ
        end
    else
        for features in eachrow(X)
            @. features = features / σ
        end
    end
end

function normalize_features!(features; center=false)
    μ, σ = compute_μ_σ_matrix(features)
    for i in eachindex(σ)
        if abs(σ[i]) < 1e-6
            σ[i] = 1.0
        end
    end
    reduce_data!(features, σ; center=center, μ=μ)
    if any(isnan, features)
        @warn("NaN values detected in features! σ = $σ")
    elseif maximum(abs.(features)) > 1e6
        @warn("some features have a very high value ! σ = $σ")
    end
end

mean_or_zero(x) = isempty(x) ? 0.0 : mean(x)
max_or_zero(x) = isempty(x) ? 0.0 : maximum(x)
min_or_zero(x) = isempty(x) ? 0.0 : minimum(x)