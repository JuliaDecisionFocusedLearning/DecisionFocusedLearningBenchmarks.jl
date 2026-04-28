has_visualization(::ContextualStochasticArgmaxBenchmark) = true

function plot_instance(::ContextualStochasticArgmaxBenchmark, sample::DataSample; kwargs...)
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

function plot_solution(::ContextualStochasticArgmaxBenchmark, sample::DataSample; kwargs...)
    x = sample.x    # full feature vector [c_base; x_raw]
    y = sample.y    # one-hot vector
    n_x = length(x)
    n = length(y)
    n_c = length(sample.c_base)

    # Color x bars: steelblue for c_base components, darkorange for x_raw components
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

    # Pick the best available utility vector to display
    if hasproperty(sample.extra, :scenario)
        u = sample.extra.scenario
        u_title = "Realized scenario ξ"
    elseif hasproperty(sample, :θ) && !isnothing(sample.θ)
        u = sample.θ
        u_title = "Predicted utilities θ̂"
    else
        u = sample.c_base
        u_title = "Base utilities c_base"
    end

    p1 = Plots.bar(
        1:n, u; legend=false, xlabel="Item", ylabel="Utility",
        title=u_title, color=:steelblue,
    )

    colors = [y[i] > 0 ? :firebrick : :steelblue for i in 1:n]
    p2 = Plots.bar(
        1:n, u; color=colors, legend=false, xlabel="Item", ylabel="Utility",
        title="Selected item (red)",
    )

    l = Plots.@layout [a{0.35h}; [b c]]
    return Plots.plot(p_x, p1, p2; layout=l, size=(800, 500), kwargs...)
end
