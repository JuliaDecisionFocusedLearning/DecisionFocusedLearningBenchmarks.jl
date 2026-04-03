has_visualization(::PortfolioOptimizationBenchmark) = true

function plot_instance(::PortfolioOptimizationBenchmark, sample::DataSample; kwargs...)
    θ = sample.θ
    d = length(θ)
    return Plots.bar(
        1:d,
        θ;
        legend=false,
        xlabel="Asset",
        ylabel="Expected return",
        title="Instance (expected returns θ)",
        color=:steelblue,
        kwargs...,
    )
end

function plot_solution(::PortfolioOptimizationBenchmark, sample::DataSample; kwargs...)
    θ = sample.θ
    y = sample.y
    d = length(θ)
    p1 = Plots.bar(
        1:d,
        θ;
        legend=false,
        xlabel="Asset",
        ylabel="Expected return",
        title="Expected returns θ",
        color=:steelblue,
    )
    p2 = Plots.bar(
        1:d,
        y;
        legend=false,
        xlabel="Asset",
        ylabel="Portfolio weight",
        title="Portfolio weights y",
        color=:seagreen,
    )
    return Plots.plot(p1, p2; layout=(1, 2), size=(800, 300), kwargs...)
end
