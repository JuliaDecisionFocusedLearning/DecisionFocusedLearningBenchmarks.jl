has_visualization(::DynamicReplenishmentBenchmark) = true

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

    if nb_customers !== nothing && !isa(nb_customers, Vector{Nothing})
        bar!(p, xs_right, -nb_customers; bar_width=w, label="No buy", color="#9a9a9a")
    end
    if sales !== nothing && !isa(sales, Vector{Nothing})
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
    n_sales=nothing,
    kwargs...,
)
    state = hasproperty(sample.context, :instance) ? sample.instance : sample.context.state
    stock = Float64.(state.stock)
    stock_p = Float64.(state.physical_stock)
    repl = Float64.(sample.y)
    sales = if hasproperty(sample.context, :next_sales)
        Float64.(sample.context.next_sales)
    else
        n_sales
    end
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
    sales=[nothing for _ in 1:length(trajectory)],
    nb_customers=[nothing for _ in 1:length(trajectory)],
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
        states = [
            hasproperty(sample.context, :instance) ? sample.instance : sample.context.state
            for sample in trajectory[steps]
        ]
        stocks = [sum(state.stock) for state in states]
        stocks_p = [sum(state.physical_stock) for state in states]
        repls = [sum(sample.y) for sample in trajectory[steps]]

        sales = if hasproperty(trajectory[1].context, :next_sales)
            [sum(sample.context.next_sales) for sample in trajectory[steps]]
        else
            sales
        end
        nb_customers = if hasproperty(trajectory[1].context, :customers)
            [sample.context.customers for sample in trajectory[steps]]
        else
            nb_customers
        end
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
                n_sales=sales[t],
                aggregated=aggregated,
                kwargs...,
            ) for t in steps
        ]
        return Plots.plot(
            plots...; layout=(rows, cols), size=(cols * 300, rows * 250), kwargs...
        )
    end
end
