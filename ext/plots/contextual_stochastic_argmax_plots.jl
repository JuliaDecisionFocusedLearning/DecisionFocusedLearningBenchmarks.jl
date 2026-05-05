has_visualization(::ContextualStochasticArgmaxBenchmark) = true

function plot_context(::ContextualStochasticArgmaxBenchmark, sample::DataSample; kwargs...)
    c_base = sample.c_base  # base utilities (first n components of x)
    x_raw = sample.x_raw    # context features (last d components of x)
    n = length(c_base)
    d = length(x_raw)

    p1 = Plots.bar(
        1:n,
        c_base;
        legend=false,
        xlabel="Item",
        ylabel="Base utility",
        title="c_base (base utilities)",
        color=:steelblue,
    )
    p2 = Plots.bar(
        1:d,
        x_raw;
        legend=false,
        xlabel="Feature",
        ylabel="Value",
        title="x_raw (context features)",
        color=:darkorange,
    )
    return Plots.plot(p1, p2; layout=(1, 2), size=(800, 300), kwargs...)
end

function plot_sample(::ContextualStochasticArgmaxBenchmark, sample::DataSample; kwargs...)
    x = sample.x
    θ = sample.θ
    y = sample.y
    n_x = length(x)
    n = length(θ)
    n_c = length(sample.c_base)

    x_colors = vcat(fill(:steelblue, n_c), fill(:darkorange, n_x - n_c))
    p_x = Plots.bar(
        1:n_x,
        x;
        color=x_colors,
        legend=false,
        xlabel="Feature index",
        ylabel="Value",
        title="x (blue = c_base, orange = x_raw)",
    )

    colors = [y[i] > 0 ? :firebrick : :steelblue for i in 1:n]
    p_θ = Plots.bar(
        1:n,
        θ;
        color=colors,
        legend=false,
        xlabel="Item",
        ylabel="Utility",
        title="θ (selected item in red)",
    )

    l = Plots.@layout [a{0.4h}; b]
    return Plots.plot(p_x, p_θ; layout=l, size=(700, 500), kwargs...)
end
