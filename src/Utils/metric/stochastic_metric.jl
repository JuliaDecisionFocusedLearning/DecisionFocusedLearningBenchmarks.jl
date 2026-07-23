# No dedicated AbstractStochasticMetric exists yet: SampleAverageApproximation is an
# AbstractStaticBenchmark (see interface/stochastic_benchmark.jl), so SAA-wrapped stochastic
# benchmarks already reuse AbstractStaticMetric / evaluate_metric from static_metric.jl as-is.
# A native (non-SAA) stochastic metric abstraction — e.g. one evaluating a per-scenario
# objective without materializing a fixed sample average — is future work.
