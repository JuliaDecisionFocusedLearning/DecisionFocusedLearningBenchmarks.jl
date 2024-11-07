var documenterSearchIndex = {"docs":
[{"location":"api/decision_focused/#Decisions-focused-learning-paper","page":"Decisions-focused learning paper","title":"Decisions-focused learning paper","text":"","category":"section"},{"location":"api/decision_focused/#Public","page":"Decisions-focused learning paper","title":"Public","text":"","category":"section"},{"location":"api/decision_focused/","page":"Decisions-focused learning paper","title":"Decisions-focused learning paper","text":"Modules = [DecisionFocusedLearningBenchmarks.FixedSizeShortestPath, DecisionFocusedLearningBenchmarks.PortfolioOptimization, DecisionFocusedLearningBenchmarks.SubsetSelection]\nPrivate = false","category":"page"},{"location":"api/decision_focused/#DecisionFocusedLearningBenchmarks.FixedSizeShortestPath.FixedSizeShortestPathBenchmark","page":"Decisions-focused learning paper","title":"DecisionFocusedLearningBenchmarks.FixedSizeShortestPath.FixedSizeShortestPathBenchmark","text":"struct FixedSizeShortestPathBenchmark <: AbstractBenchmark\n\nBenchmark problem for the shortest path problem. In this benchmark, all graphs are acyclic directed grids, all of the same size grid_size. Features are given at instance level (one dimensional vector of length p for each graph).\n\nData is generated using the process described in: https://arxiv.org/abs/2307.13565.\n\nFields\n\ngraph::Graphs.SimpleGraphs.SimpleDiGraph{Int64}: grid graph instance\ngrid_size::Tuple{Int64, Int64}: grid size of graphs\np::Int64: size of feature vectors\ndeg::Int64: degree of formula between features and true weights\nν::Float32: multiplicative noise for true weights sampled between [1-ν, 1+ν], should be between 0 and 1\n\n\n\n\n\n","category":"type"},{"location":"api/decision_focused/#DecisionFocusedLearningBenchmarks.FixedSizeShortestPath.FixedSizeShortestPathBenchmark-Tuple{}","page":"Decisions-focused learning paper","title":"DecisionFocusedLearningBenchmarks.FixedSizeShortestPath.FixedSizeShortestPathBenchmark","text":"FixedSizeShortestPathBenchmark(\n;\n    grid_size,\n    p,\n    deg,\n    ν\n) -> FixedSizeShortestPathBenchmark\n\n\nConstructor for FixedSizeShortestPathBenchmark.\n\n\n\n\n\n","category":"method"},{"location":"api/decision_focused/#DecisionFocusedLearningBenchmarks.PortfolioOptimization.PortfolioOptimizationBenchmark","page":"Decisions-focused learning paper","title":"DecisionFocusedLearningBenchmarks.PortfolioOptimization.PortfolioOptimizationBenchmark","text":"struct PortfolioOptimizationBenchmark <: AbstractBenchmark\n\nBenchmark problem for the portfolio optimization problem.\n\nData is generated using the process described in: https://arxiv.org/abs/2307.13565.\n\nFields\n\nd::Int64: number of assets\np::Int64: size of feature vectors\ndeg::Int64: hypermarameter for data generation\nν::Float32: another hyperparameter, should be positive\nΣ::Matrix{Float32}: covariance matrix\nγ::Float32: maximum variance of portfolio\nL::Matrix{Float32}: useful for dataset generation\nf::Vector{Float32}: useful for dataset generation\n\n\n\n\n\n","category":"type"},{"location":"api/decision_focused/#DecisionFocusedLearningBenchmarks.PortfolioOptimization.PortfolioOptimizationBenchmark-Tuple{}","page":"Decisions-focused learning paper","title":"DecisionFocusedLearningBenchmarks.PortfolioOptimization.PortfolioOptimizationBenchmark","text":"PortfolioOptimizationBenchmark(\n;\n    d,\n    p,\n    deg,\n    ν,\n    seed\n) -> PortfolioOptimizationBenchmark\n\n\nConstructor for PortfolioOptimizationBenchmark.\n\n\n\n\n\n","category":"method"},{"location":"api/decision_focused/#DecisionFocusedLearningBenchmarks.SubsetSelection.SubsetSelectionBenchmark","page":"Decisions-focused learning paper","title":"DecisionFocusedLearningBenchmarks.SubsetSelection.SubsetSelectionBenchmark","text":"struct SubsetSelectionBenchmark <: AbstractBenchmark\n\nBenchmark problem for the subset selection problem. Reference: https://arxiv.org/abs/2307.13565.\n\nThe goal is to select the best k items from a set of n items, without knowing their values, but only observing some features.\n\nFields\n\nn::Int64: total number of items\nk::Int64: number of items to select\n\n\n\n\n\n","category":"type"},{"location":"api/decision_focused/#Private","page":"Decisions-focused learning paper","title":"Private","text":"","category":"section"},{"location":"api/decision_focused/","page":"Decisions-focused learning paper","title":"Decisions-focused learning paper","text":"Modules = [DecisionFocusedLearningBenchmarks.FixedSizeShortestPath, DecisionFocusedLearningBenchmarks.PortfolioOptimization, DecisionFocusedLearningBenchmarks.SubsetSelection]\nPublic = false","category":"page"},{"location":"api/decision_focused/#DecisionFocusedLearningBenchmarks.Utils.generate_dataset","page":"Decisions-focused learning paper","title":"DecisionFocusedLearningBenchmarks.Utils.generate_dataset","text":"generate_dataset(\n    bench::FixedSizeShortestPathBenchmark;\n    ...\n) -> Vector\ngenerate_dataset(\n    bench::FixedSizeShortestPathBenchmark,\n    dataset_size::Int64;\n    seed,\n    type\n) -> Vector\n\n\nGenerate dataset for the shortest path problem.\n\n\n\n\n\n","category":"function"},{"location":"api/decision_focused/#DecisionFocusedLearningBenchmarks.Utils.generate_maximizer-Tuple{FixedSizeShortestPathBenchmark}","page":"Decisions-focused learning paper","title":"DecisionFocusedLearningBenchmarks.Utils.generate_maximizer","text":"generate_maximizer(\n    bench::FixedSizeShortestPathBenchmark;\n    use_dijkstra\n) -> DecisionFocusedLearningBenchmarks.FixedSizeShortestPath.var\"#shortest_path_maximizer#8\"{DecisionFocusedLearningBenchmarks.FixedSizeShortestPath.var\"#shortest_path_maximizer#5#9\"{typeof(Graphs.dijkstra_shortest_paths), Vector{Int64}, Vector{Int64}, Int64, Int64, Graphs.SimpleGraphs.SimpleDiGraph{Int64}}}\n\n\nOutputs a function that computes the longest path on the grid graph, given edge weights θ as input.\n\nmaximizer = generate_maximizer(bench)\nmaximizer(θ)\n\n\n\n\n\n","category":"method"},{"location":"api/decision_focused/#DecisionFocusedLearningBenchmarks.Utils.generate_statistical_model-Tuple{FixedSizeShortestPathBenchmark}","page":"Decisions-focused learning paper","title":"DecisionFocusedLearningBenchmarks.Utils.generate_statistical_model","text":"generate_statistical_model(\n    bench::FixedSizeShortestPathBenchmark\n) -> Flux.Chain{T} where T<:Tuple{Flux.Dense{typeof(identity), Matrix{Float32}}}\n\n\nInitialize a linear model for bench using Flux.\n\n\n\n\n\n","category":"method"},{"location":"api/decision_focused/#DecisionFocusedLearningBenchmarks.Utils.generate_dataset-2","page":"Decisions-focused learning paper","title":"DecisionFocusedLearningBenchmarks.Utils.generate_dataset","text":"generate_dataset(\n    bench::PortfolioOptimizationBenchmark;\n    ...\n) -> Vector\ngenerate_dataset(\n    bench::PortfolioOptimizationBenchmark,\n    dataset_size::Int64;\n    seed,\n    type\n) -> Vector\n\n\nGenerate a dataset of labeled instances for the portfolio optimization problem.\n\n\n\n\n\n","category":"function"},{"location":"api/decision_focused/#DecisionFocusedLearningBenchmarks.Utils.generate_maximizer-Tuple{PortfolioOptimizationBenchmark}","page":"Decisions-focused learning paper","title":"DecisionFocusedLearningBenchmarks.Utils.generate_maximizer","text":"generate_maximizer(\n    bench::PortfolioOptimizationBenchmark\n) -> DecisionFocusedLearningBenchmarks.PortfolioOptimization.var\"#portfolio_maximizer#3\"{Float32, Matrix{Float32}, Int64}\n\n\nCreate a function solving the MIQP formulation of the portfolio optimization problem.\n\n\n\n\n\n","category":"method"},{"location":"api/decision_focused/#DecisionFocusedLearningBenchmarks.Utils.generate_statistical_model-Tuple{PortfolioOptimizationBenchmark}","page":"Decisions-focused learning paper","title":"DecisionFocusedLearningBenchmarks.Utils.generate_statistical_model","text":"generate_statistical_model(\n    bench::PortfolioOptimizationBenchmark\n) -> Flux.Dense{typeof(identity), Matrix{Float32}}\n\n\nInitialize a linear model for bench using Flux.\n\n\n\n\n\n","category":"method"},{"location":"api/decision_focused/#DecisionFocusedLearningBenchmarks.Utils.generate_dataset-3","page":"Decisions-focused learning paper","title":"DecisionFocusedLearningBenchmarks.Utils.generate_dataset","text":"generate_dataset(\n    bench::SubsetSelectionBenchmark;\n    ...\n) -> Any\ngenerate_dataset(\n    bench::SubsetSelectionBenchmark,\n    dataset_size::Int64;\n    seed,\n    identity_mapping\n) -> Any\n\n\nGenerate a dataset of labeled instances for the subset selection problem. The mapping between features and cost is identity.\n\n\n\n\n\n","category":"function"},{"location":"api/decision_focused/#DecisionFocusedLearningBenchmarks.Utils.generate_maximizer-Tuple{SubsetSelectionBenchmark}","page":"Decisions-focused learning paper","title":"DecisionFocusedLearningBenchmarks.Utils.generate_maximizer","text":"generate_maximizer(\n    bench::SubsetSelectionBenchmark\n) -> Base.Fix2{typeof(DecisionFocusedLearningBenchmarks.SubsetSelection.top_k), Int64}\n\n\nReturn a top k maximizer.\n\n\n\n\n\n","category":"method"},{"location":"api/decision_focused/#DecisionFocusedLearningBenchmarks.Utils.generate_statistical_model-Tuple{SubsetSelectionBenchmark}","page":"Decisions-focused learning paper","title":"DecisionFocusedLearningBenchmarks.Utils.generate_statistical_model","text":"generate_statistical_model(\n    bench::SubsetSelectionBenchmark;\n    seed\n) -> Flux.Dense{typeof(identity), Matrix{Float32}}\n\n\nInitialize a linear model for bench using Flux.\n\n\n\n\n\n","category":"method"},{"location":"api/interface/#Interface","page":"Interface","title":"Interface","text":"","category":"section"},{"location":"api/interface/#Public","page":"Interface","title":"Public","text":"","category":"section"},{"location":"api/interface/","page":"Interface","title":"Interface","text":"Modules = [DecisionFocusedLearningBenchmarks.Utils]\nPrivate = false","category":"page"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.AbstractBenchmark","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.AbstractBenchmark","text":"abstract type AbstractBenchmark\n\nAbstract type interface for a benchmark problem.\n\nThe following methods exist for benchmarks:\n\ngenerate_dataset\ngenerate_statistical_model\ngenerate_maximizer\nplot_data\nobjective_value\ncompute_gap\n\n\n\n\n\n","category":"type"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.DataSample","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.DataSample","text":"struct DataSample{F, S, C, I}\n\nData sample data structure.\n\nFields\n\nx::Any: features\nθ::Any: costs\ny::Any: solution\ninstance::Any: instance\n\n\n\n\n\n","category":"type"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.average_tensor-Tuple{Any}","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.average_tensor","text":"average_tensor(x)\n\nAverage the tensor x along its third axis.\n\n\n\n\n\n","category":"method"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.compute_gap","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.compute_gap","text":"compute_gap(::AbstractBenchmark, dataset::Vector{<:DataSample}, statistical_model, maximizer) -> Float64\n\nCompute the average relative optimality gap of the pipeline on the dataset.\n\n\n\n\n\n","category":"function"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.compute_gap-Tuple{AbstractBenchmark, Vector{<:DataSample}, Any, Any}","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.compute_gap","text":"compute_gap(\n    bench::AbstractBenchmark,\n    dataset::Vector{<:DataSample},\n    statistical_model,\n    maximizer\n) -> Any\n\n\nDefault behaviour of compute_gap for a benchmark problem where features, solutions and costs are all defined.\n\n\n\n\n\n","category":"method"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.generate_dataset","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.generate_dataset","text":"generate_dataset(::AbstractBenchmark, dataset_size::Int) -> Vector{<:DataSample}\n\nGenerate a Vector of DataSample  of length dataset_size for given benchmark. Content of the dataset can be visualized using plot_data, when it applies.\n\n\n\n\n\n","category":"function"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.generate_maximizer","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.generate_maximizer","text":"generate_maximizer(::AbstractBenchmark)\n\nGenerates a maximizer function. Returns a callable f: (θ; kwargs...) -> y, where θ is a cost array and y is a solution.\n\n\n\n\n\n","category":"function"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.generate_statistical_model","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.generate_statistical_model","text":"generate_statistical_model(::AbstractBenchmark)\n\nInitializes and return an untrained statistical model of the CO-ML pipeline. It's usually a Flux model, that takes a feature matrix x as iinput, and returns a cost array θ as output.\n\n\n\n\n\n","category":"function"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.get_path-Tuple{AbstractVector{<:Integer}, Integer, Integer}","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.get_path","text":"get_path(\n    parents::AbstractVector{<:Integer},\n    s::Integer,\n    d::Integer\n) -> Vector{T} where T<:Integer\n\n\nRetrieve a path from the parents array and start sand endd`` of path.\n\n\n\n\n\n","category":"method"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.grid_graph-Union{Tuple{AbstractMatrix{R}}, Tuple{R}} where R","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.grid_graph","text":"grid_graph(\n    costs::AbstractArray{R, 2};\n    acyclic\n) -> SimpleWeightedGraphs.SimpleWeightedDiGraph{Int64}\n\n\nConvert a grid of cell costs into a weighted directed graph from SimpleWeightedGraphs.jl, where the vertices correspond to the cells and the edges are weighted by the cost of the arrival cell.\n\nIf acyclic = false, a cell has edges to each one of its 8 neighbors.\nIf acyclic = true, a cell has edges to its south, east and southeast neighbors only (ensures an acyclic graph where topological sort will work)\n\nThis can be used to model the Warcraft shortest paths problem of\n\nDifferentiation of Blackbox Combinatorial Solvers, Vlastelica et al. (2019)\n\n\n\n\n\n","category":"method"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.neg_tensor-Tuple{Any}","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.neg_tensor","text":"neg_tensor(x)\n\nCompute minus softplus element-wise on tensor x.\n\n\n\n\n\n","category":"method"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.path_to_matrix-Tuple{Vector{<:Integer}, Integer, Integer}","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.path_to_matrix","text":"path_to_matrix(\n    path::Vector{<:Integer},\n    h::Integer,\n    w::Integer\n) -> Matrix{Int64}\n\n\nTransform path into a binary matrix of size (h, w) where each cell is 1 if the cell is part of the path, 0 otherwise.\n\n\n\n\n\n","category":"method"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.plot_data","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.plot_data","text":"plot_data(::AbstractBenchmark, args...)\n\nPlot a data sample from the dataset created by generate_dataset. Check the specific benchmark documentation of plot_data for more details on the arguments.\n\n\n\n\n\n","category":"function"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.squeeze_last_dims-Tuple{Any}","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.squeeze_last_dims","text":"squeeze_last_dims(x)\n\nSqueeze two last dimensions on tensor x.\n\n\n\n\n\n","category":"method"},{"location":"api/interface/#Private","page":"Interface","title":"Private","text":"","category":"section"},{"location":"api/interface/","page":"Interface","title":"Interface","text":"Modules = [DecisionFocusedLearningBenchmarks.Utils]\nPublic = false","category":"page"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.coord_to_index-NTuple{4, Integer}","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.coord_to_index","text":"coord_to_index(\n    i::Integer,\n    j::Integer,\n    h::Integer,\n    w::Integer\n) -> Any\n\n\nGiven a pair of row-column coordinates (i, j) on a grid of size (h, w), compute the corresponding vertex index in the graph generated by grid_graph.\n\n\n\n\n\n","category":"method"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.count_edges-Tuple{Integer, Integer}","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.count_edges","text":"count_edges(h::Integer, w::Integer; acyclic)\n\n\nCompute the number of edges in a grid graph of size (h, w).\n\n\n\n\n\n","category":"method"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.index_to_coord-Tuple{Integer, Integer, Integer}","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.index_to_coord","text":"index_to_coord(\n    v::Integer,\n    h::Integer,\n    w::Integer\n) -> Tuple{Any, Any}\n\n\nGiven a vertex index in the graph generated by grid_graph, compute the corresponding row-column coordinates (i, j) on a grid of size (h, w).\n\n\n\n\n\n","category":"method"},{"location":"api/interface/#DecisionFocusedLearningBenchmarks.Utils.objective_value-Tuple{AbstractBenchmark, AbstractArray, AbstractArray}","page":"Interface","title":"DecisionFocusedLearningBenchmarks.Utils.objective_value","text":"objective_value(\n    _::AbstractBenchmark,\n    θ̄::AbstractArray,\n    y::AbstractArray\n) -> Any\n\n\nDefault behaviour of objective_value.\n\n\n\n\n\n","category":"method"},{"location":"benchmarks/fixed_size_shortest_path/#Shortest-paths","page":"Shortest paths","title":"Shortest paths","text":"","category":"section"},{"location":"benchmarks/fixed_size_shortest_path/","page":"Shortest paths","title":"Shortest paths","text":"FixedSizeShortestPathBenchmark is a benchmark problem that consists of finding the shortest path in a grid graph between the top left and bottom right corners. In this benchmark, the grid size is the same for all instances.","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"EditURL = \"tutorials/warcraft.jl\"","category":"page"},{"location":"warcraft/#Path-finding-on-image-maps","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"","category":"section"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"In this tutorial, we showcase DecisionFocusedLearningBenchmarks.jl capabilities on one of its main benchmarks: the Warcraft benchmark. This benchmark problem is a simple path-finding problem where the goal is to find the shortest path between the top left and bottom right corners of a given image map. The map is represented as a 2D image representing a 12x12 grid, each cell having an unknown travel cost depending on the terrain type.","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"First, let's load the package and create a benchmark object as follows:","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"using DecisionFocusedLearningBenchmarks\nb = WarcraftBenchmark()","category":"page"},{"location":"warcraft/#Dataset-generation","page":"Path-finding on image maps","title":"Dataset generation","text":"","category":"section"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"These benchmark objects behave as generators that can generate various needed elements in order to build an algorithm to tackle the problem. First of all, all benchmarks are capable of generating datasets as needed, using the generate_dataset method. This method takes as input the benchmark object for which the dataset is to be generated, and a second argument specifying the number of samples to generate:","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"dataset = generate_dataset(b, 50);\nnothing #hide","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"We obtain a vector of DataSample objects, containing all needed data for the problem. Subdatasets can be created through regular slicing:","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"train_dataset, test_dataset = dataset[1:45], dataset[46:50]","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"And getting an individual sample will return a DataSample with four fields: x, instance, θ, and y:","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"sample = test_dataset[1]","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"x correspond to the input features, i.e. the input image (3D array) in the Warcraft benchmark case:","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"x = sample.x","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"θ correspond to the true unknown terrain weights. We use the opposite of the true weights in order to formulate the optimization problem as a maximization problem:","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"θ_true = sample.θ","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"y correspond to the optimal shortest path, encoded as a binary matrix:","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"y_true = sample.y","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"instance is not used in this benchmark, therefore set to nothing:","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"isnothing(sample.instance)","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"For some benchmarks, we provide the following plotting method plot_data to visualize the data:","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"plot_data(b, sample)","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"We can see here the terrain image, the true terrain weights, and the true shortest path avoiding the high cost cells.","category":"page"},{"location":"warcraft/#Building-a-pipeline","page":"Path-finding on image maps","title":"Building a pipeline","text":"","category":"section"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"DecisionFocusedLearningBenchmarks also provides methods to build an hybrid machine learning and combinatorial optimization pipeline for the benchmark. First, the generate_statistical_model method generates a machine learning predictor to predict cell weights from the input image:","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"model = generate_statistical_model(b)","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"In the case of the Warcraft benchmark, the model is a convolutional neural network built using the Flux.jl package.","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"θ = model(x)","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"Note that the model is not trained yet, and its parameters are randomly initialized.","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"Finally, the generate_maximizer method can be used to generate a combinatorial optimization algorithm that takes the predicted cell weights as input and returns the corresponding shortest path:","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"maximizer = generate_maximizer(b; dijkstra=true)","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"In the case o fthe Warcraft benchmark, the method has an additional keyword argument to chose the algorithm to use: Dijkstra's algorithm or Bellman-Ford algorithm.","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"y = maximizer(θ)","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"As we can see, currently the pipeline predicts random noise as cell weights, and therefore the maximizer returns a straight line path.","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"plot_data(b, DataSample(; x, θ, y))","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"We can evaluate the current pipeline performance using the optimality gap metric:","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"starting_gap = compute_gap(b, test_dataset, model, maximizer)","category":"page"},{"location":"warcraft/#Using-a-learning-algorithm","page":"Path-finding on image maps","title":"Using a learning algorithm","text":"","category":"section"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"We can now train the model using the InferOpt.jl package:","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"using InferOpt\nusing Flux\nusing Plots\n\nperturbed_maximizer = PerturbedMultiplicative(maximizer; ε=0.2, nb_samples=100)\nloss = FenchelYoungLoss(perturbed_maximizer)\n\nstarting_gap = compute_gap(b, test_dataset, model, maximizer)\n\nopt_state = Flux.setup(Adam(1e-3), model)\nloss_history = Float64[]\nfor epoch in 1:50\n    val, grads = Flux.withgradient(model) do m\n        sum(loss(m(sample.x), sample.y) for sample in train_dataset) / length(train_dataset)\n    end\n    Flux.update!(opt_state, model, grads[1])\n    push!(loss_history, val)\nend\n\nplot(loss_history; xlabel=\"Epoch\", ylabel=\"Loss\", title=\"Training loss\")","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"final_gap = compute_gap(b, test_dataset, model, maximizer)","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"θ = model(x)\ny = maximizer(θ)\nplot_data(b, DataSample(; x, θ, y))","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"","category":"page"},{"location":"warcraft/","page":"Path-finding on image maps","title":"Path-finding on image maps","text":"This page was generated using Literate.jl.","category":"page"},{"location":"api/warcraft/#Warcraft","page":"Warcraft","title":"Warcraft","text":"","category":"section"},{"location":"api/warcraft/#Public","page":"Warcraft","title":"Public","text":"","category":"section"},{"location":"api/warcraft/","page":"Warcraft","title":"Warcraft","text":"Modules = [DecisionFocusedLearningBenchmarks.Warcraft]\nPrivate = false","category":"page"},{"location":"api/warcraft/#DecisionFocusedLearningBenchmarks.Warcraft.WarcraftBenchmark","page":"Warcraft","title":"DecisionFocusedLearningBenchmarks.Warcraft.WarcraftBenchmark","text":"struct WarcraftBenchmark <: AbstractBenchmark\n\nBenchmark for the Warcraft shortest path problem. Does not have any field.\n\n\n\n\n\n","category":"type"},{"location":"api/warcraft/#Private","page":"Warcraft","title":"Private","text":"","category":"section"},{"location":"api/warcraft/","page":"Warcraft","title":"Warcraft","text":"Modules = [DecisionFocusedLearningBenchmarks.Warcraft]\nPublic = false","category":"page"},{"location":"api/warcraft/#DecisionFocusedLearningBenchmarks.Utils.generate_dataset","page":"Warcraft","title":"DecisionFocusedLearningBenchmarks.Utils.generate_dataset","text":"generate_dataset(::WarcraftBenchmark) -> Vector\ngenerate_dataset(\n    ::WarcraftBenchmark,\n    dataset_size::Int64\n) -> Vector\n\n\nDownloads and decompresses the Warcraft dataset the first time it is called.\n\nwarning: Warning\ndataset_size is capped at 10000, i.e. the number of available samples in the dataset files.\n\n\n\n\n\n","category":"function"},{"location":"api/warcraft/#DecisionFocusedLearningBenchmarks.Utils.generate_maximizer-Tuple{WarcraftBenchmark}","page":"Warcraft","title":"DecisionFocusedLearningBenchmarks.Utils.generate_maximizer","text":"generate_maximizer(\n    ::WarcraftBenchmark;\n    dijkstra\n) -> typeof(DecisionFocusedLearningBenchmarks.Warcraft.dijkstra_maximizer)\n\n\nReturns an optimization algorithm that computes a longest path on the grid graph with given weights. Uses a shortest path algorithm on opposite weights to get the longest path.\n\n\n\n\n\n","category":"method"},{"location":"api/warcraft/#DecisionFocusedLearningBenchmarks.Utils.generate_statistical_model-Tuple{WarcraftBenchmark}","page":"Warcraft","title":"DecisionFocusedLearningBenchmarks.Utils.generate_statistical_model","text":"generate_statistical_model(\n    ::WarcraftBenchmark;\n    seed\n) -> Flux.Chain{T} where T<:Tuple{Any, Any, Any, Any, Flux.AdaptiveMaxPool{4, 2}, typeof(DecisionFocusedLearningBenchmarks.Utils.average_tensor), typeof(DecisionFocusedLearningBenchmarks.Utils.neg_tensor), typeof(DecisionFocusedLearningBenchmarks.Utils.squeeze_last_dims)}\n\n\nCreate and return a Flux.Chain embedding for the Warcraft terrains, inspired by differentiation of blackbox combinatorial solvers.\n\nThe embedding is made as follows:\n\nThe first 5 layers of ResNet18 (convolution, batch normalization, relu, maxpooling and first resnet block).\nAn adaptive maxpooling layer to get a (12x12x64) tensor per input image.\nAn average over the third axis (of size 64) to get a (12x12x1) tensor per input image.\nThe element-wize neg_tensor function to get cell weights of proper sign to apply shortest path algorithms.\nA squeeze function to forget the two last dimensions.\n\n\n\n\n\n","category":"method"},{"location":"api/warcraft/#DecisionFocusedLearningBenchmarks.Utils.plot_data-Tuple{WarcraftBenchmark, DataSample}","page":"Warcraft","title":"DecisionFocusedLearningBenchmarks.Utils.plot_data","text":"plot_data(\n    ::WarcraftBenchmark,\n    sample::DataSample;\n    θ_true,\n    θ_title,\n    y_title,\n    kwargs...\n) -> Any\n\n\nPlot the content of input DataSample as images. x as the initial image, θ as the weights, and y as the path.\n\nThe keyword argument θ_true is used to set the color range of the weights plot.\n\n\n\n\n\n","category":"method"},{"location":"api/warcraft/#DecisionFocusedLearningBenchmarks.Warcraft.bellman_maximizer-Tuple{AbstractMatrix}","page":"Warcraft","title":"DecisionFocusedLearningBenchmarks.Warcraft.bellman_maximizer","text":"bellman_maximizer(\n    θ::AbstractMatrix;\n    kwargs...\n) -> Matrix{Int64}\n\n\nComputes the longest path in given grid graph weights by computing the shortest path in the graph with opposite weights. Using the Ford-Bellman dynamic programming algorithm.\n\n\n\n\n\n","category":"method"},{"location":"api/warcraft/#DecisionFocusedLearningBenchmarks.Warcraft.convert_image_for_plot-Tuple{Array{Float32, 3}}","page":"Warcraft","title":"DecisionFocusedLearningBenchmarks.Warcraft.convert_image_for_plot","text":"convert_image_for_plot(\n    image::Array{Float32, 3}\n) -> Matrix{ColorTypes.RGB{FixedPointNumbers.N0f8}}\n\n\nConvert image to the proper data format to enable plots in Julia.\n\n\n\n\n\n","category":"method"},{"location":"api/warcraft/#DecisionFocusedLearningBenchmarks.Warcraft.create_dataset-Tuple{String, Int64}","page":"Warcraft","title":"DecisionFocusedLearningBenchmarks.Warcraft.create_dataset","text":"create_dataset(\n    decompressed_path::String,\n    nb_samples::Int64\n) -> Vector\n\n\nCreate the dataset corresponding to the data located at decompressed_path, possibly sub-sampling nb_samples points. The dataset is made of images of Warcraft terrains, cell cost labels and shortest path labels. It is a Vector of tuples, each Tuple being a dataset point.\n\n\n\n\n\n","category":"method"},{"location":"api/warcraft/#DecisionFocusedLearningBenchmarks.Warcraft.dijkstra_maximizer-Tuple{AbstractMatrix}","page":"Warcraft","title":"DecisionFocusedLearningBenchmarks.Warcraft.dijkstra_maximizer","text":"dijkstra_maximizer(\n    θ::AbstractMatrix;\n    kwargs...\n) -> Matrix{Int64}\n\n\nComputes the longest path in given grid graph weights by computing the shortest path in the graph with opposite weights. Using the Dijkstra algorithm.\n\nwarning: Warning\nOnly works on graph with positive weights, i.e. if θ only contains negative values.\n\n\n\n\n\n","category":"method"},{"location":"api/warcraft/#DecisionFocusedLearningBenchmarks.Warcraft.grid_bellman_ford_warcraft","page":"Warcraft","title":"DecisionFocusedLearningBenchmarks.Warcraft.grid_bellman_ford_warcraft","text":"grid_bellman_ford_warcraft(g, s, d, length_max)\n\nApply the Bellman-Ford algorithm on an GridGraph g, and return a ShortestPathTree with source s and destination d, among the paths having length smaller than length_max.\n\n\n\n\n\n","category":"function"},{"location":"api/warcraft/#DecisionFocusedLearningBenchmarks.Warcraft.read_dataset","page":"Warcraft","title":"DecisionFocusedLearningBenchmarks.Warcraft.read_dataset","text":"read_dataset(\n    decompressed_path::String\n) -> Tuple{Any, Any, Any}\nread_dataset(\n    decompressed_path::String,\n    dtype::String\n) -> Tuple{Any, Any, Any}\n\n\nRead the dataset of type dtype at the decompressed_path location. The dataset is made of images of Warcraft terrains, cell cost labels and shortest path labels. They are returned separately, with proper axis permutation and image scaling to be consistent with Flux embeddings.\n\n\n\n\n\n","category":"function"},{"location":"benchmarks/warcraft/#Warcraft","page":"Warcraft","title":"Warcraft","text":"","category":"section"},{"location":"benchmarks/warcraft/","page":"Warcraft","title":"Warcraft","text":"See the tutorial for a full demo of WarcraftBenchmark.","category":"page"},{"location":"benchmarks/subset_selection/#Subset-Selection","page":"Subset Selection","title":"Subset Selection","text":"","category":"section"},{"location":"benchmarks/subset_selection/","page":"Subset Selection","title":"Subset Selection","text":"SubsetSelectionBenchmark is the most trivial benchmark problem in this package. It is minimalistic and serves as a simple example for debugging and testing purposes.","category":"page"},{"location":"benchmarks/subset_selection/#Description","page":"Subset Selection","title":"Description","text":"","category":"section"},{"location":"benchmarks/subset_selection/","page":"Subset Selection","title":"Subset Selection","text":"We have a set of n items, each item having an unknown value. We want to select a subset of k items that maximizes the sum of the values of the selected items.","category":"page"},{"location":"benchmarks/subset_selection/","page":"Subset Selection","title":"Subset Selection","text":"As input, instead of the items costs, we are given a feature vector, such that an unknown linear mapping between the feature vector and the value of the items exists.","category":"page"},{"location":"benchmarks/subset_selection/","page":"Subset Selection","title":"Subset Selection","text":"By default, this linear mapping is the identity mapping, i.e., the value of each item is equal to the value of the corresponding feature vector element. However, this mapping can be changed by setting the identity_mapping parameter to false.","category":"page"},{"location":"benchmarks/portfolio_optimization/#Portfolio-Optimization","page":"Portfolio Optimization","title":"Portfolio Optimization","text":"","category":"section"},{"location":"benchmarks/portfolio_optimization/","page":"Portfolio Optimization","title":"Portfolio Optimization","text":"PortfolioOptimizationBenchmark is a Markovitz portfolio optimization problem, where asset prices are unknown, and only contextual data is available to predict these prices. The goal is to predict asset prices c and maximize the expected return of a portfolio, subject to a risk constraint using this maximization program:","category":"page"},{"location":"benchmarks/portfolio_optimization/","page":"Portfolio Optimization","title":"Portfolio Optimization","text":"beginaligned\nmaxquad  c^top x\ntextstquad  x^top Sigma x leq gamma\n 1^top x leq 1\n x geq 0\nendaligned","category":"page"},{"location":"#DecisionFocusedLearningBenchmarks.jl","page":"Home","title":"DecisionFocusedLearningBenchmarks.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Dev) (Image: Build Status) (Image: Coverage) (Image: Code Style: Blue)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Set of benchmark problems to be solved with DecisionFocusedLearning.jl","category":"page"}]
}
