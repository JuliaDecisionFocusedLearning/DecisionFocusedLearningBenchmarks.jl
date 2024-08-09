@testitem "Subset selection" begin
    using InferOptBenchmarks
    using InferOpt
    using Flux
    using UnicodePlots
    using Zygote

    b = SubsetSelectionBenchmark()

    dataset = generate_dataset(b, 500)
    model = generate_statistical_model(b)
    maximizer = generate_maximizer(b)

    train_dataset, test_dataset = dataset[1:450], dataset[451:500]
    X_train = train_dataset.features
    Y_train = train_dataset.solutions

    perturbed_maximizer = PerturbedAdditive(maximizer; ε=1.0, nb_samples=100)
    loss = FenchelYoungLoss(perturbed_maximizer)

    starting_gap = compute_gap(b, test_dataset, model, maximizer)

    opt_state = Flux.setup(Adam(0.1), model)
    loss_history = Float64[]
    for epoch in 1:50
        val, grads = Flux.withgradient(model) do m
            sum(loss(m(x), y) for (x, y) in zip(X_train, Y_train)) / length(train_dataset)
        end
        Flux.update!(opt_state, model, grads[1])
        push!(loss_history, val)
    end

    final_gap = compute_gap(b, test_dataset, model, maximizer)

    lineplot(loss_history)
    @test loss_history[end] < loss_history[1]
    @test final_gap < starting_gap / 10
end
