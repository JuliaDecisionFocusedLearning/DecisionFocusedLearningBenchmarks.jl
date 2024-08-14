# # Path-finding on image maps

using InferOptBenchmarks
using InferOpt
using Flux
using Plots

b = WarcraftBenchmark()

dataset = generate_dataset(b, 50)
model = generate_statistical_model(b)
maximizer = generate_maximizer(b)

train_dataset, test_dataset = dataset[1:45], dataset[46:50]
X_train = train_dataset.features
Y_train = train_dataset.solutions

perturbed_maximizer = PerturbedMultiplicative(maximizer; ε=0.2, nb_samples=100)
loss = FenchelYoungLoss(perturbed_maximizer)

starting_gap = compute_gap(b, test_dataset, model, maximizer)

opt_state = Flux.setup(Adam(1e-3), model)
loss_history = Float64[]
for epoch in 1:50
    val, grads = Flux.withgradient(model) do m
        sum(loss(m(x), y) for (x, y) in zip(X_train, Y_train)) / length(train_dataset)
    end
    Flux.update!(opt_state, model, grads[1])
    push!(loss_history, val)
end

plot(loss_history; xlabel="Epoch", ylabel="Loss", title="Training loss")

#

final_gap = compute_gap(b, test_dataset, model, maximizer)