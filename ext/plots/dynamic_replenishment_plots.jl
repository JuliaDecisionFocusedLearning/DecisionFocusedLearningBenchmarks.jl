has_visualization(::DynamicReplenishmentBenchmark) = true

"""
Bar plot of stock level of each items.
"""
function plot_sample(
    b::DynamicReplenishmentBenchmark,
    sample::DataSample;
    with_legend=true,
    with_title=true,
    n_sales=nothing,
    kwargs...,
)
    RB = DecisionFocusedLearningBenchmarks.DynamicReplenishment
    state = hasproperty(sample.context, :instance) ? sample.instance : sample.context.state
    N = RB.item_count(state.config)

    stock = Float64.(state.stock)
    repl = Float64.(RB.get_replenishment_from_y(sample.y; state=state))
    # check if extra field is present, otherwise use next_sales argument

    sales = if hasproperty(sample.context, :next_sales)
        Float64.(sample.context.next_sales)
    else
        n_sales
    end
    println(sales)
    if sales !== nothing
        sales = -sales
        w = 0.5
        xs_left = (1:N) .- w/2
        xs_right = (1:N) .+ w/2
    else
        w = 1
        xs_left = 1:N
    end

    legend = with_legend ? :topleft : false
    title = with_title ? "Stock, replenishment and sales" : ""

    p = bar(
        xs_left,
        stock .+ repl;
        bar_width=w,
        label="Replenishment",
        color="#1baf7a",   # vert pour la barre totale (repl visible en haut)
        xlabel="Item",
        ylabel="Count",
        title=title,
        legend=legend,
        xticks=1:N,
        size=(800, 500),
    )
    bar!(p, xs_left, stock; bar_width=w, label="Stock", color="#2a78d6")
    if sales !== nothing
        bar!(p, xs_right, sales; bar_width=w, label="Sales", color="#e34948")
    end
    return p
end

function plot_trajectory(
    bench::DynamicReplenishmentBenchmark,
    trajectory::Vector{<:DataSample};
    sales=[nothing for _ in 1:length(trajectory)],
    max_steps=10,
    cols=3,
    kwargs...,
)
    n = min(length(trajectory), max_steps)
    rows = ceil(Int, n / cols)
    steps = round.(Int, range(1, length(trajectory); length=n))
    upper_middle = div(cols, 2) + 1
    plots = [
        plot_sample(
            bench,
            trajectory[t];
            with_legend=(t == 1),
            with_title=(t == upper_middle),
            n_sales=sales[t],
            kwargs...,
        ) for t in steps
    ]
    return Plots.plot(
        plots...; layout=(rows, cols), size=(cols * 300, rows * 250), kwargs...
    )
end