"""
$TYPEDSIGNATURES

Create a `VSPInstance` from file `filepath` containing a VRPTW instance.
It uses time window values to compute task times as the middle of the interval.

Round all values to `Int` if `rounded=true`.
Normalize all time values by the `normalization` parameter.
"""
function read_vsp_instance(filepath::String; normalization=3600.0, digits=2)
    type = Float64 #rounded ? Int : Float64
    mode = ""
    edge_weight_type = ""
    edge_weight_format = ""
    duration_matrix = Vector{type}[]
    nb_locations = 0
    demand = type[]
    service_time = type[]
    coordinates = Matrix{type}(undef, 0, 2)
    start_time = type[]

    file = open(filepath, "r")
    for line in eachline(file)
        line = strip(line, [' ', '\n', '\t'])
        if line == ""
            continue
        elseif startswith(line, "DIMENSION")
            nb_locations = parse(Int, split(line, " : ")[2])
            demand = zeros(type, nb_locations)
            service_time = zeros(type, nb_locations)
            coordinates = zeros(type, (nb_locations, 2))
            start_time = zeros(type, nb_locations)
        elseif startswith(line, "EDGE_WEIGHT_TYPE")
            edge_weight_type = split(line, " : ")[2]
        elseif startswith(line, "EDGE_WEIGHT_FORMAT")
            edge_weight_format = split(line, " : ")[2]
        elseif startswith(line, "NODE_COORD_SECTION")
            mode = "coord"
        elseif line == "DEMAND_SECTION"
            mode = "demand"
        elseif line == "DEPOT_SECTION"
            mode = "depot"
        elseif line == "EDGE_WEIGHT_SECTION"
            mode = "edge_weights"
            @assert edge_weight_type == "EXPLICIT"
            @assert edge_weight_format == "FULL_MATRIX"
        elseif line == "TIME_WINDOW_SECTION"
            mode = "time_windows"
        elseif line == "SERVICE_TIME_SECTION"
            mode = "service_t"
        elseif line == "EOF"
            break
        elseif mode == "coord"
            node, x, y = split(line)  # Split by whitespace or \t, skip duplicate whitespace
            node = parse(Int, node)
            x, y = (parse(type, x), parse(type, y))
            coordinates[node, :] = [x, y]
        elseif mode == "demand"
            node, d = split(line)
            node, d = parse(Int, node), parse(type, d)
            if node == 1 # depot
                @assert d == 0
            end
            demand[node] = d
        elseif mode == "edge_weights"
            push!(duration_matrix, [parse(type, e) for e in split(line)])
        elseif mode == "service_t"
            node, t = split(line)
            node = parse(Int, node)
            t = parse(type, t)
            if node == 1 # depot
                @assert t == 0
            end
            service_time[node] = t
        elseif mode == "time_windows"
            node, l, u = split(line)
            node = parse(Int, node)
            l, u = parse(type, l), parse(type, u)
            start_time[node] = (u + l) / 2
        end
    end
    close(file)

    duration = mapreduce(permutedims, vcat, duration_matrix)

    coordinate = [
        Point(round(x / normalization; digits), round(y / normalization; digits)) for
        (x, y) in zip(coordinates[:, 1], coordinates[:, 2])
    ]
    service_time ./= normalization
    start_time ./= normalization
    duration ./= normalization

    return StaticInstance(;
        coordinate,
        service_time=round.(service_time; digits),
        start_time=round.(start_time; digits),
        duration=round.(duration; digits),
    )
end
