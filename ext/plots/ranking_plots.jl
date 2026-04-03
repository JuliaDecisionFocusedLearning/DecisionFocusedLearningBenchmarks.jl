has_visualization(::RankingBenchmark) = true

function plot_instance(::RankingBenchmark, sample::DataSample; kwargs...)
    θ = sample.θ
    n = length(θ)
    return Plots.bar(
        1:n,
        θ;
        legend=false,
        xlabel="Item",
        ylabel="Cost",
        title="Instance (costs θ)",
        color=:steelblue,
        kwargs...,
    )
end

function plot_solution(::RankingBenchmark, sample::DataSample; kwargs...)
    θ = sample.θ
    y = sample.y  # y[i] = rank of item i (1 = best)
    n = length(θ)
    # Color by rank: rank 1 (best) in dark blue, rank n (worst) in light
    palette = Plots.cgrad(:Blues, n; rev=true, categorical=true)
    colors = [palette[y[i]] for i in 1:n]
    return Plots.bar(
        1:n,
        θ;
        legend=false,
        xlabel="Item",
        ylabel="Cost",
        title="Solution (color = rank, dark = best)",
        color=colors,
        kwargs...,
    )
end
