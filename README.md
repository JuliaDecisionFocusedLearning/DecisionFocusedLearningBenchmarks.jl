# DecisionFocusedLearningBenchmarks.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/DecisionFocusedLearningBenchmarks.jl/dev/)
[![Build Status](https://github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningBenchmarks.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningBenchmarks.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaDecisionFocusedLearning/DecisionFocusedLearningBenchmarks.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/JuliaDecisionFocusedLearning/DecisionFocusedLearningBenchmarks.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

This repository contains a collection of benchmark problems for decision-focused learning algorithms.
It provides a common interface for creating datasets, associated statistical models and combinatorial optimization maximizers for building decision-focused learning pipelines.
They can be used for instance as benchmarks for tools in [InferOpt.jl](https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl), but can be used in any other context as well.

Currently, this package provides the following benchmark problems (many more to come!):
- `SubsetSelectionBenchmark`: a minimalist subset selection problem.
- `FixedSizeShortestPathBenchmark`: shortest path problem with on a graph with fixed size.
- `WarcraftBenchmark`: shortest path problem on image maps
- `PortfolioOptimizationBenchmark`: portfolio optimization problem.

See the [documentation](https://JuliaDecisionFocusedLearning.github.io/DecisionFocusedLearningBenchmarks.jl/stable/) for more details.
