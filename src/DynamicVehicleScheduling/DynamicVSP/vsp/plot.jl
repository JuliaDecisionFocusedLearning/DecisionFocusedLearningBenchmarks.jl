"""
$TYPEDSIGNATURES

Plot the given static VSP `instance`.
"""
function plot_instance(
    instance::VSPInstance;
    customer_markersize=4,
    depot_markersize=7,
    alpha_depot=0.8,
    customer_color=:lightblue,
    depot_color=:lightgreen,
    kwargs...,
)
    x = [p.x for p in instance.coordinate]
    y = [p.y for p in instance.coordinate]

    fig = plot(; legend=:topleft, xlabel="x coordinate", ylabel="y coordinate", kwargs...)
    scatter!(
        fig,
        x[2:end],
        y[2:end];
        label="Customers",
        markercolor=customer_color,
        marker=:circle,
        markersize=customer_markersize,
    )
    scatter!(
        fig,
        [x[1]],
        [y[1]];
        label="Depot",
        markercolor=depot_color,
        marker=:rect,
        markersize=depot_markersize,
        alpha=alpha_depot,
    )
    return fig
end
