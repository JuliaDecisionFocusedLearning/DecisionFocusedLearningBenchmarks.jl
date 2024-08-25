# # Path-finding on image maps

#=
In this tutorial, we showcase InferOptBenchmarks.jl capabilities on one of its main benchmarks: the Warcraft benchmark.
This benchmark problem is a simple path-finding problem where the goal is to find the shortest path between the top left and bottom right corners of a given image map.
The map is represented as a 2D image representing a 12x12 grid, each cell having an unknown travel cost depending on the terrain type.
=#

# First, let's load the package and create a benchmark object as follows:
using InferOptBenchmarks
b = WarcraftBenchmark()

# ## Dataset generation

# These benchmark objects behave as generators that can generate various needed elements in order to build an algorithm to tackle the problem.
# First of all, all benchmarks are capable of generating datasets as needed, using the [`generate_dataset`](@ref) method.
# This method takes as input the benchmark object for which the dataset is to be generated, and a second argument specifying the number of samples to generate:
dataset = generate_dataset(b, 50)

# We obtain an [`InferOptDataset`](@ref) object, which contains all needed data for the problem.
# Subdatasets can be created through regular slicing:
train_dataset, test_dataset = dataset[1:45], dataset[46:50]

# And getting an individual sample will return a NamedTuple with four keys: `features`, `instance`, `costs`, and `solution`:
sample = test_dataset[1]
# In the case the the Warcraft benchmark, `features` correspond to the input image:
x = sample.features
# `costs` correspond to the true unknown terrain weights:
θ_true = sample.costs
# `solution` correspond to the true shortest path:
y_true = sample.solution
# `instance` is not used in this benchmark, therefore set to nothing:
sample.instance

# For some benchmarks, we provide the following plotting method [`plot_data`](@ref) to visualize the data:
plot_data(b; sample...)
# We can see here the terrain image, the true terrain weights, and the true shortest path avoiding the high cost cells.

# ## Building a pipeline

# InferOptBenchmarks also provides methods to build an hybrid machine learning and combinatorial optimization pipeline for the benchmark.
# First, the [`generate_statistical_model`](@ref) method generates a machine learning predictor to predict cell weights from the input image:
model = generate_statistical_model(b)
# In the case of the Warcraft benchmark, the model is a convolutional neural network built using the Flux.jl package.
θ = model(x)
# Note that the model is not trained yet, and its parameters are randomly initialized.

# Finally, the [`generate_maximizer`](@ref) method can be used generates a combinatorial optimization algorithm that takes the predicted cell weights as input and returns the corresponding shortest path:
maximizer = generate_maximizer(b; dijkstra=true)
# In the case o fthe Warcraft benchmark, the method has an additioonal keyword argument to chose the algorithm to use: Dijkstra's algorithm or Bellman-Ford algorithm.
y = maximizer(θ)
# As we can see, currently the pipeline predicts random noise as cell weights, and therefore the maximizer returns a straight line path.
plot_data(b; features=x, costs=θ, solution=y)
# We can evaluate the current pipeline performance using the optimality gap metric:
starting_gap = compute_gap(b, test_dataset, model, maximizer)

# ## Using a learning algorithm

# We can now train the model using the InferOpt.jl package:
using InferOpt
using Flux
using Plots

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

#
plot_data(b; sample...)
