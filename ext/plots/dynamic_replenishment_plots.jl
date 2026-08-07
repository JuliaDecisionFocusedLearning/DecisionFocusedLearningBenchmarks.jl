has_visualization(::DynamicReplenishmentBenchmark) = true

function plot_context(bench::DynamicReplenishmentBenchmark, sample::DataSample; kwargs...)
    static_u = bench.static_utilities[1:(end - 1)]  # drop the "no purchase" option
    state = sample.state
    stock = Float64.(state.stock)
    stock_p = Float64.(state.physical_stock)
    N = length(stock)

    p1 = bar(
        1:N,
        stock;
        label="Virtual stock",
        color="#5fd6a8",
        ylabel="Count",
        title="Stock levels",
        xticks=(1:N, fill("", N)),
    )
    bar!(p1, 1:N, stock_p; label="Physical stock", color="#0f7d52")

    p2 = bar(
        1:N,
        static_u;
        legend=false,
        xlabel="Item",
        ylabel="Utility",
        title="Static utilities",
        color="#2a78d6",
    )

    l = Plots.@layout [a{0.6h}; b{0.4h}]
    return Plots.plot(p1, p2; layout=l, size=(800, 600), kwargs...)
end

function bar_plot_stock_repl_sales(
    stock,
    stock_p,
    repl,
    sales=nothing,
    nb_customers=nothing;
    xlabel::String="Item",
    ylabel::String="Count",
    title::String="Stock, replenishment and sales",
    legend,
    kwargs...,
)
    xmax = length(stock)
    if sales !== nothing || nb_customers !== nothing
        w = 0.5
        xs_left = (1:xmax) .- w/2
        xs_right = (1:xmax) .+ w/2
    else
        w = 1
        xs_left = 1:xmax
    end

    p = bar(
        xs_left,
        stock .+ repl;
        bar_width=w,
        label="Replenishment",
        color="#2a78d6", # blue
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        legend=legend,
        xticks=1:xmax,
        size=(800, 500),
    )
    bar!(p, xs_left, stock; bar_width=w, label="Virtual stock", color="#5fd6a8")
    bar!(p, xs_left, stock_p; bar_width=w, label="Physical stock", color="#0f7d52")

    if nb_customers !== nothing
        bar!(p, xs_right, -nb_customers; bar_width=w, label="No buy", color="#9a9a9a")
    end
    if sales !== nothing
        bar!(p, xs_right, -sales; bar_width=w, label="Sales", color="#e34948")
    end
    return p
end

"""
Bar plot of stock level of each items.
"""
function plot_sample(
    b::DynamicReplenishmentBenchmark,
    sample::DataSample;
    with_legend=true,
    with_title=true,
    kwargs...,
)
    state = sample.state
    stock = Float64.(state.stock)
    stock_p = Float64.(state.physical_stock)
    repl = Float64.(sample.y)
    sales = Float64.(sample.next_sales)
    return p = bar_plot_stock_repl_sales(
        stock,
        stock_p,
        repl,
        sales,
        nothing;
        xlabel="Item",
        ylabel="Count",
        title=with_title ? "Stock, replenishment and sales" : "",
        legend=with_legend ? :topright : false,
        kwargs...,
    )
end

function plot_trajectory(
    bench::DynamicReplenishmentBenchmark,
    trajectory::Vector{<:DataSample};
    max_steps=10,
    cols=3,
    aggregated::Bool=false,
    kwargs...,
)
    n = min(length(trajectory), max_steps)
    rows = ceil(Int, n / cols)
    steps = round.(Int, range(1, length(trajectory); length=n))
    upper_middle = div(cols, 2) + 1
    if aggregated
        states = [sample.state for sample in trajectory[steps]]
        stocks = [sum(state.stock) for state in states]
        stocks_p = [sum(state.physical_stock) for state in states]
        repls = [sum(sample.y) for sample in trajectory[steps]]
        sales = [sum(sample.next_sales) for sample in trajectory[steps]]
        nb_customers = [sample.customers for sample in trajectory[steps]]
        return bar_plot_stock_repl_sales(
            stocks,
            stocks_p,
            repls,
            sales,
            nb_customers;
            xlabel="Time step",
            ylabel="Count",
            title="Total stock, replenishment and sales over time",
            legend=:topright,
            kwargs...,
        )
    else
        plots = [
            plot_sample(
                bench,
                trajectory[t];
                with_legend=(t == 1),
                with_title=(t == upper_middle),
                kwargs...,
            ) for t in steps
        ]
        return Plots.plot(
            plots...; layout=(rows, cols), size=(cols * 300, rows * 250), kwargs...
        )
    end
end
