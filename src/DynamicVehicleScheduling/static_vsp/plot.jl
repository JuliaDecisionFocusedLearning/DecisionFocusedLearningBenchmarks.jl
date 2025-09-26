function plot_instance(
    x_depot,
    y_depot,
    x_customers,
    y_customers;
    customer_markersize=4,
    depot_markersize=7,
    alpha_depot=0.8,
    customer_color=:lightblue,
    depot_color=:lightgreen,
    kwargs...,
)
    fig = plot(; legend=:topleft, xlabel="x coordinate", ylabel="y coordinate", kwargs...)
    scatter!(
        fig,
        x_customers,
        y_customers;
        label="Customers",
        markercolor=customer_color,
        marker=:circle,
        markersize=customer_markersize,
    )
    scatter!(
        fig,
        [x_depot],
        [y_depot];
        label="Depot",
        markercolor=depot_color,
        marker=:rect,
        markersize=depot_markersize,
        alpha=alpha_depot,
    )
    return fig
end

function build_instance_data(instance::StaticInstance)
    x = [p.x for p in instance.coordinate]
    y = [p.y for p in instance.coordinate]
    return (x_depot=x[1], y_depot=y[1], x_customers=x[2:end], y_customers=y[2:end])
end

"""
$TYPEDSIGNATURES

Plot the given static VSP `instance`.
"""
function plot_instance(instance::StaticInstance; kwargs...)
    x_depot, y_depot, x, y = build_instance_data(instance)

    return plot_instance(x_depot, y_depot, x, y; kwargs...)
end
