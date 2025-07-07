function oracle(θ; instance::DVSPState, kwargs...)
    routes = prize_collecting_vsp(θ; instance=instance, kwargs...)
    return VSPSolution(
        routes; max_index=location_count(instance.state_instance)
    ).edge_matrix
end

function g(y; instance, kwargs...)
    return vec(sum(y[:, instance.is_postponable]; dims=1))
end

function h(y, duration)
    value = 0.0
    N = size(duration, 1)
    for i in 1:N
        for j in 1:N
            value -= y[i, j] * duration[i, j]
        end
    end
    return value
end

function h(y; instance, kwargs...)
    return h(y, instance.state_instance.duration)
end
