"""
$TYPEDEF

Abstract type for evaluation metrics on **stochastic** benchmarks.
A stochastic metric is a function of a single sample along with a scenario: implement `(m::MyStochasticMetric)(sample::DataSample) -> Real`
and [`metric_benchmark`](@ref). [`evaluate_metric`](@ref) maps it over the samples of a dataset, one value per instance.

Note : For now, the package handles the evaluation of stochastic benchmarks via SampleAverageApproximation which is an AbstractStaticBenchmark (see interface/stochastic_benchmark.jl), so SAA-wrapped stochastic
benchmarks already reuse AbstractStaticMetric / evaluate_metric from static_metric.jl as-is.
=> A native stochastic metric abstraction — e.g. one evaluating a per-scenario
objective without materializing a fixed sample average — is future work.
"""
abstract type AbstractStochasticMetric{B<:AbstractStochasticBenchmark} <: AbstractMetric{B} end
