has_visualization(::ContextualStochasticArgmaxBenchmark) = true

function plot_instance(::ContextualStochasticArgmaxBenchmark, sample::DataSample; kwargs...)
    c_base = sample.c_base  # base utilities from context
    n = length(c_base)
    return Plots.bar(
        1:n,
        c_base;
        legend=false,
        xlabel="Item",
        ylabel="Base utility",
        title="Instance (base utilities c_base)",
        color=:steelblue,
        kwargs...,
    )
end

function plot_solution(::ContextualStochasticArgmaxBenchmark, sample::DataSample; kwargs...)
    y = sample.y  # one-hot vector
    n = length(y)

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
        1:n,
        u;
        legend=false,
        xlabel="Item",
        ylabel="Utility",
        title=u_title,
        color=:steelblue,
    )

    colors = [y[i] > 0 ? :firebrick : :steelblue for i in 1:n]
    p2 = Plots.bar(
        1:n,
        u;
        color=colors,
        legend=false,
        xlabel="Item",
        ylabel="Utility",
        title="Selected item (red)",
    )

    return Plots.plot(p1, p2; layout=(1, 2), size=(800, 300), kwargs...)
end
