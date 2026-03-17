function build_instance_data(instance::StaticInstance)
    x = [p.x for p in instance.coordinate]
    y = [p.y for p in instance.coordinate]
    return (x_depot=x[1], y_depot=y[1], x_customers=x[2:end], y_customers=y[2:end])
end
