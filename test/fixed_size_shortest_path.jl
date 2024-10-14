module FixedSizeShortestPathTest

using DecisionFocusedLearningBenchmarks.FixedSizeShortestPath

bench = FixedSizeShortestPathBenchmark()

(; features, costs, solutions) = generate_dataset(bench)

model = generate_statistical_model(bench)
maximizer = generate_maximizer(bench)

end
