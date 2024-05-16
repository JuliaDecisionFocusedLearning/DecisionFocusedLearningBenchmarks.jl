using InferOptBenchmarks.SubsetSelection

bench = SubsetSelectionBenchmark()

(; features, solutions) = generate_dataset(bench, 1000)
model = generate_statistical_model(bench)
