# # Path-finding on image maps

#=
In this tutorial, we showcase DecisionFocusedLearningBenchmarks.jl capabilities on one of its main benchmarks: the Warcraft benchmark.
This benchmark problem is a simple path-finding problem where the goal is to find the shortest path between the top left and bottom right corners of a given image map.
The map is represented as a 2D image representing a 12x12 grid, each cell having an unknown travel cost depending on the terrain type.
=#

# First, let's load the package and create a benchmark object as follows:
using DecisionFocusedLearningBenchmarks
using Plots
b = WarcraftBenchmark()

# ## Dataset generation

# These benchmark objects behave as generators that can generate various needed elements in order to build an algorithm to tackle the problem.
# First of all, all benchmarks are capable of generating datasets as needed, using the [`generate_dataset`](@ref) method.
# This method takes as input the benchmark object for which the dataset is to be generated, and a second argument specifying the number of samples to generate:
dataset = generate_dataset(b, 50);

# We obtain a vector of [`DataSample`](@ref) objects, containing all needed data for the problem.
# Subdatasets can be created through regular slicing:
train_dataset, test_dataset = dataset[1:45], dataset[46:50]

# And getting an individual sample will return a [`DataSample`](@ref) with five fields: `x`, `θ`, `y`, `context`, and `extra`:
sample = test_dataset[1]
# `x` correspond to the input features, i.e. the input image (3D array) in the Warcraft benchmark case:
x = sample.x
# `θ` correspond to the true unknown terrain weights. We use the opposite of the true weights in order to formulate the optimization problem as a maximization problem:
θ_true = sample.θ
# `y` correspond to the optimal shortest path, encoded as a binary matrix:
y_true = sample.y
# `context` is not used in this benchmark (no solver kwargs needed), so it is empty:
isempty(sample.context)
# `extra` is also not used in this benchmark, so it is empty as well:
isempty(sample.extra)

# For some benchmarks, we provide the following plotting method [`plot_sample`](@ref) to visualize the data:
plot_sample(b, sample)
# We can see here the terrain image, the true terrain weights, and the true shortest path avoiding the high cost cells.

# ## Building a pipeline

# DecisionFocusedLearningBenchmarks also provides methods to build a hybrid machine learning and combinatorial optimization pipeline for the benchmark.
# First, the [`generate_statistical_model`](@ref) method generates a Lux model architecture.
# The convenience method with an RNG argument returns the model together with initialized parameters and state:
using Lux, StableRNGs

rng = StableRNG(0)
model, ps, st = generate_statistical_model(b, rng)
model
# In the case of the Warcraft benchmark, the model is a convolutional neural network built using Lux.jl.
st_test = Lux.testmode(st)
θ, _ = model(x, ps, st_test)
# Note that the model is not trained yet, and its parameters are randomly initialized.

# Finally, the [`generate_maximizer`](@ref) method can be used to generate a combinatorial optimization algorithm that takes the predicted cell weights as input and returns the corresponding shortest path:
maximizer = generate_maximizer(b; dijkstra=true)
# In the case of the Warcraft benchmark, the method has an additional keyword argument to choose the algorithm to use: Dijkstra's algorithm or Bellman-Ford algorithm.
y = maximizer(θ)
# As we can see, currently the pipeline predicts random noise as cell weights, and therefore the maximizer returns a straight line path.
plot_sample(b, DataSample(; x, θ, y))
# We can evaluate the current pipeline performance using the optimality gap metric:
starting_gap = compute_gap(b, test_dataset, model, ps, st, maximizer)

# ## Using a learning algorithm

# We can now train the model using the InferOpt.jl package.
# InferOpt is framework-agnostic (it uses ChainRulesCore), so its layers and losses work identically with Lux.
using InferOpt

perturbed_maximizer = PerturbedMultiplicative(maximizer; ε=0.2, nb_samples=100)
loss = FenchelYoungLoss(perturbed_maximizer)

# We use Lux's `Training` API, which manages parameters and optimizer state via a `TrainState`.
# The objective function must follow the signature `(model, ps, st, data) -> (loss_value, updated_st, stats)`.
# We wrap our InferOpt loss in a callable struct to make it compatible with this interface:

struct LuxLoss{L}
    loss::L
end

function (obj::LuxLoss)(model, ps, st, (x, y))
    θ_pred, st = model(x, ps, st)
    return obj.loss(θ_pred, y), st, (;)
end

lux_loss = LuxLoss(loss)

# We wrap the training loop in a function and run it:
using Zygote
using Optimisers: Adam

function train_loop(model, ps, st, lux_loss, train_dataset; epochs=50)
    train_state = Lux.Training.TrainState(model, ps, st, Adam(1e-3))
    loss_history = Float64[]
    for epoch in 1:epochs
        epoch_loss = 0.0
        for (; x, y) in train_dataset
            _, l, _, train_state = Lux.Training.single_train_step!(
                AutoZygote(), lux_loss, (x, y), train_state
            )
            epoch_loss += l
        end
        push!(loss_history, epoch_loss / length(train_dataset))
    end
    return train_state.parameters, train_state.states, loss_history
end

ps, st, loss_history = train_loop(model, ps, st, lux_loss, train_dataset)

plot(loss_history; xlabel="Epoch", ylabel="Loss", title="Training loss")

#

final_gap = compute_gap(b, test_dataset, model, ps, st, maximizer)

#
st_test = Lux.testmode(st)
θ, _ = model(x, ps, st_test)
y = maximizer(θ)
plot_sample(b, DataSample(; x, θ, y))

using Test #src
@test final_gap < starting_gap #src
