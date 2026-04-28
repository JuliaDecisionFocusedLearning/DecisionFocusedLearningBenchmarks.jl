has_visualization(::PortfolioOptimizationBenchmark) = true

function plot_instance(::PortfolioOptimizationBenchmark, sample::DataSample; kwargs...)
    x = sample.x
    p = length(x)
    return Plots.bar(
        1:p,
        Float64.(x);
        legend=false,
        xlabel="Feature",
        ylabel="Value",
        title="Features x (observable input)",
        color=:steelblue,
        xticks=1:p,
        kwargs...,
    )
end

function plot_solution(::PortfolioOptimizationBenchmark, sample::DataSample; kwargs...)
    x = sample.x
    θ = sample.θ
    y = sample.y
    p = length(x)
    d = length(θ)

    p_x = Plots.bar(
        1:p, Float64.(x);
        legend=false, xlabel="Feature", ylabel="Value",
        title="x (features, observable)", color=:steelblue, xticks=1:p,
    )
    p1 = Plots.bar(
        1:d, θ;
        legend=false, xlabel="Asset", ylabel="Expected return",
        title="θ (expected returns)", color=:steelblue,
    )
    p2 = Plots.bar(
        1:d, y;
        legend=false, xlabel="Asset", ylabel="Portfolio weight",
        title="y (portfolio weights)", color=:seagreen,
    )

    l = Plots.@layout [a{0.3h}; [b c]]
    return Plots.plot(p_x, p1, p2; layout=l, size=(800, 500), kwargs...)
end
