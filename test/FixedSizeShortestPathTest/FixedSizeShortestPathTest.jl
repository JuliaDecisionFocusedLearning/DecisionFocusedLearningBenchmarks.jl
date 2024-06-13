module FixedSizeShortestPathTest

using InferOptBenchmarks.FixedSizeShortestPath

using Flux
using InferOpt
using ProgressMeter
using UnicodePlots
using Zygote

bench = FixedSizeShortestPathBenchmark()

(; features, costs, solutions) = generate_dataset(bench)

model = generate_statistical_model(bench)
maximizer = generate_maximizer(bench)

perturbed = PerturbedAdditive(maximizer; nb_samples=10, ε=0.1)
fyl = FenchelYoungLoss(perturbed)

opt_state = Flux.setup(Adam(), model)
loss_history = Float64[]
gap_history = Float64[]
E = 100
@showprogress for epoch in 1:E
    loss = 0.0
    for (x, y) in zip(features, solutions)
        val, grads = Flux.withgradient(model) do m
            θ = m(x)
            fyl(θ, y)
        end
        loss += val
        Flux.update!(opt_state, model, grads[1])
    end
    push!(loss_history, loss ./ E)
    push!(
        gap_history, compute_gap(bench, model, features, costs, solutions, maximizer) .* 100
    )
end

println(lineplot(loss_history; title="Loss"))
println(lineplot(gap_history; title="Gap"))

end
