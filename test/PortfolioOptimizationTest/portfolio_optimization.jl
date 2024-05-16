using InferOptBenchmarks.PortfolioOptimization

using Flux
using InferOpt
using ProgressMeter
using UnicodePlots
using Zygote

bench = PortfolioOptimizationBenchmark()

(; features, costs, solutions) = generate_dataset(bench, 1000)
model = generate_statistical_model(bench)
maximizer = generate_maximizer(bench)

x = features[1]
y = solutions[1]
θ = model(x)
y_pred = maximizer(θ)

maximum(y_pred)
