function _init_plot(title="")
    pl = Plots.plot(;
        aspect_ratio=:equal,
        legend=:outerleft,
        xlim=(-1.1, 1.1),
        ylim=(-1.1, 1.1),
        title=title,
    )
    return pl
end

function _plot_polytope!(pl, vertices)
    return Plots.plot!(
        pl,
        vcat(map(first, vertices), first(vertices[1])),
        vcat(map(last, vertices), last(vertices[1]));
        fillrange=0,
        fillcolor=:gray,
        fillalpha=0.2,
        linecolor=:black,
        label=L"\mathrm{conv}(\mathcal{Y}(x))",
    )
end

function _plot_objective!(pl, θ)
    Plots.plot!(
        pl, [0.0, θ[1]], [0.0, θ[2]]; color="#9558B2", arrow=true, lw=2, label=nothing
    )
    Plots.annotate!(pl, [-0.2 * θ[1]], [-0.2 * θ[2]], [L"\theta"])
    return pl
end

function _plot_y!(pl, y)
    return Plots.scatter!(
        pl,
        [y[1]],
        [y[2]];
        color="#CB3C33",
        markersize=9,
        markershape=:square,
        label=L"f(\theta)",
    )
end

function _plot_maximizer!(pl, θ, instance, maximizer)
    ŷ = maximizer(θ; instance)
    return _plot_y!(pl, ŷ)
end

has_visualization(::Argmax2DBenchmark) = true

function plot_instance(::Argmax2DBenchmark, sample::DataSample)
    pl = _init_plot()
    _plot_polytope!(pl, sample.instance)
    return pl
end

function plot_solution(::Argmax2DBenchmark, sample::DataSample)
    pl = _init_plot()
    _plot_polytope!(pl, sample.instance)
    _plot_objective!(pl, sample.θ)
    return _plot_y!(pl, sample.y)
end

function plot_solution(::Argmax2DBenchmark, sample::DataSample, y; θ=sample.θ)
    pl = _init_plot()
    _plot_polytope!(pl, sample.instance)
    _plot_objective!(pl, θ)
    return _plot_y!(pl, y)
end
