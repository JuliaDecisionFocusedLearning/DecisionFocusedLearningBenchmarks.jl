function build_polytope(N; shift=0.0)
    return [[cospi(2k / N + shift), sinpi(2k / N + shift)] for k in 0:(N - 1)]
end

function init_plot(title="")
    pl = Plots.plot(;
        aspect_ratio=:equal,
        legend=:outerleft,
        xlim=(-1.1, 1.1),
        ylim=(-1.1, 1.1),
        title=title,
    )
    return pl
end;

function plot_polytope!(pl, vertices)
    return Plots.plot!(
        vcat(map(first, vertices), first(vertices[1])),
        vcat(map(last, vertices), last(vertices[1]));
        fillrange=0,
        fillcolor=:gray,
        fillalpha=0.2,
        linecolor=:black,
        label=L"\mathrm{conv}(\mathcal{Y}(x))",
    )
end;

function plot_objective!(pl, θ)
    Plots.plot!(
        pl,
        [0.0, θ[1]],
        [0.0, θ[2]];
        color=Colors.JULIA_LOGO_COLORS.purple,
        arrow=true,
        lw=2,
        label=nothing,
    )
    Plots.annotate!(pl, [-0.2 * θ[1]], [-0.2 * θ[2]], [L"\theta"])
    return pl
end;

function plot_maximizer!(pl, θ, instance, maximizer)
    ŷ = maximizer(θ; instance)
    return Plots.scatter!(
        pl,
        [ŷ[1]],
        [ŷ[2]];
        color=Colors.JULIA_LOGO_COLORS.red,
        markersize=9,
        markershape=:square,
        label=L"f(\theta)",
    )
end;
