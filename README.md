# DecisionFocusedLearningBenchmarks.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/DecisionFocusedLearningBenchmarks.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/DecisionFocusedLearningBenchmarks.jl/dev/)
[![Build Status](https://github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningBenchmarks.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningBenchmarks.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaDecisionFocusedLearning/DecisionFocusedLearningBenchmarks.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/JuliaDecisionFocusedLearning/DecisionFocusedLearningBenchmarks.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

> [!WARNING]
>  This package is currently under active development. The API may change in future releases.

**DecisionFocusedLearningBenchmarks.jl** provides a collection of benchmark problems for evaluating [Decision-Focused Learning](https://JuliaDecisionFocusedLearning.github.io/DecisionFocusedLearningBenchmarks.jl/stable/) algorithms, spanning static, stochastic, and dynamic settings.
Each benchmark provides a dataset generator, a statistical model architecture, and a combinatorial oracle, ready to plug into any DFL training algorithm:

```julia
using DecisionFocusedLearningBenchmarks

bench = ArgmaxBenchmark()
dataset = generate_dataset(bench, 100)
model = generate_statistical_model(bench)
maximizer = generate_maximizer(bench)
```

For the full list of benchmarks, the common interface, and detailed usage examples, refer to the [documentation](https://JuliaDecisionFocusedLearning.github.io/DecisionFocusedLearningBenchmarks.jl/stable/).

## Installation

```julia
using Pkg
Pkg.add("DecisionFocusedLearningBenchmarks")
```

## Related Packages

This package is part of the [JuliaDecisionFocusedLearning](https://github.com/JuliaDecisionFocusedLearning) organization, and built to be compatible with other packages in the ecosystem:
- **[InferOpt.jl](https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl)**: differentiable optimization layers and losses for decision-focused learning
- **[DecisionFocusedLearningAlgorithms.jl](https://github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningAlgorithms.jl)**: collection of generic black-box implementations of decision-focused learning algorithms
