@testitem "Warcraft" begin
    using InferOptBenchmarks
    using InferOpt
    using Flux
    using Zygote

    b = WarcraftBenchmark()

    dataset = generate_dataset(b, 50)
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)

    train_dataset, test_dataset = dataset[1:45], dataset[46:50]
    X_train = train_dataset.features
    Y_train = train_dataset.solutions

    x, y_true, θ_true = test_dataset[1]
    θ = model(x)

    perturbed_maximizer = PerturbedMultiplicative(maximizer; ε=0.2, nb_samples=100)
    loss = FenchelYoungLoss(perturbed_maximizer)

    yp = perturbed_maximizer(-θ_true)

    opt_state = Flux.setup(Adam(1e-3), model)
    loss_history = Float64[]
    for epoch in 1:50
        val, grads = Flux.withgradient(model) do m
            sum(loss(m(x), y) for (x, y) in zip(X_train, Y_train)) / length(train_dataset)
        end
        Flux.update!(opt_state, model, grads[1])
        push!(loss_history, val)
    end

    @test loss_history[end] < loss_history[1]
end
