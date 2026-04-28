import DecisionFocusedLearningBenchmarks.Warcraft as W
using Images: Gray

has_visualization(::WarcraftBenchmark) = true

function plot_instance(::WarcraftBenchmark, sample::DataSample; kwargs...)
    im = dropdims(sample.x; dims=4)
    img = W.convert_image_for_plot(im)
    return Plots.plot(
        img; aspect_ratio=:equal, framestyle=:none, title="Terrain image", kwargs...
    )
end

function plot_solution(
    ::WarcraftBenchmark,
    sample::DataSample;
    θ_true=sample.θ,
    θ_title="Cell costs θ",
    y_title="Path y",
    kwargs...,
)
    x = sample.x
    y = sample.y
    θ = sample.θ
    im = dropdims(x; dims=4)
    img = W.convert_image_for_plot(im)
    p1 = Plots.plot(
        img; aspect_ratio=:equal, framestyle=:none, size=(300, 300), title="Terrain image"
    )
    p2 = Plots.heatmap(
        -θ;
        yflip=true,
        aspect_ratio=:equal,
        framestyle=:none,
        padding=(0.0, 0.0),
        size=(300, 300),
        legend=false,
        title=θ_title,
        clim=(minimum(-θ_true), maximum(-θ_true)),
    )
    p3 = Plots.plot(
        Gray.(y .* 0.7);
        aspect_ratio=:equal,
        framestyle=:none,
        size=(300, 300),
        title=y_title,
    )
    return Plots.plot(p1, p2, p3; layout=(1, 3), size=(900, 300), kwargs...)
end
