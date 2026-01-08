# DecisionFocusedLearningBenchmarks.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/DecisionFocusedLearningBenchmarks.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/DecisionFocusedLearningBenchmarks.jl/dev/)
[![Build Status](https://github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningBenchmarks.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningBenchmarks.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaDecisionFocusedLearning/DecisionFocusedLearningBenchmarks.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/JuliaDecisionFocusedLearning/DecisionFocusedLearningBenchmarks.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

!!! warning 
    This package is currently under active development. The API may change in future releases.
    Please refer to the [documentation](https://JuliaDecisionFocusedLearning.github.io/DecisionFocusedLearningBenchmarks.jl/stable/) for the latest updates.

## What is Decision-Focused Learning?

Decision-Focused Learning (DFL) is a paradigm that integrates machine learning prediction with combinatorial optimization to make better decisions under uncertainty.
Unlike traditional "predict-then-optimize" approaches that optimize prediction accuracy independently of downstream decision quality, DFL directly optimizes end-to-end decision performance.

A typical DFL algorithm involves training a parametrized policy that combines a statistical predictor with an optimization component:

```math
x \;\longrightarrow\; \boxed{\,\text{Statistical model } \varphi_w\,} 
\;\xrightarrow{\theta}\; \boxed{\,\text{CO algorithm } f\,} 
\;\longrightarrow\; y
```

Where:
- **Statistical model** $\varphi_w$: machine learning predictor (e.g., neural network)
- **CO algorithm** $f$: combinatorial optimization solver
- **Instance** $x$: input data (e.g., features, context)
- **Parameters** $\theta$: predicted parameters for the optimization problem solved by `f`
- **Solution** $y$: output decision/solution

## Package Overview

**DecisionFocusedLearningBenchmarks.jl** provides a comprehensive collection of benchmark problems for evaluating decision-focused learning algorithms. The package offers:

- **Standardized benchmark problems** spanning diverse application domains
- **Common interfaces** for creating datasets, statistical models, and optimization algorithms
- **Ready-to-use DFL policies** compatible with [InferOpt.jl](https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl) and the whole [JuliaDecisionFocusedLearning](https://github.com/JuliaDecisionFocusedLearning) ecosystem
- **Evaluation tools** for comparing algorithm performance

## Benchmark Categories

The package organizes benchmarks into three main categories based on their problem structure:

### Static Benchmarks (`AbstractBenchmark`)
Single-stage optimization problems with no randomness involved:
- [`ArgmaxBenchmark`](@ref): argmax toy problem
- [`Argmax2DBenchmark`](@ref): 2D argmax toy problem
- [`RankingBenchmark`](@ref): ranking problem
- [`SubsetSelectionBenchmark`](@ref): select optimal subset of items
- [`PortfolioOptimizationBenchmark`](@ref): portfolio optimization problem
- [`FixedSizeShortestPathBenchmark`](@ref): find shortest path on grid graphs with fixed size
- [`WarcraftBenchmark`](@ref): shortest path on image maps

### Stochastic Benchmarks (`AbstractStochasticBenchmark`)  
Single-stage optimization problems under uncertainty:
- [`StochasticVehicleSchedulingBenchmark`](@ref): stochastic vehicle scheduling under delay uncertainty

### Dynamic Benchmarks (`AbstractDynamicBenchmark`)
Multi-stage sequential decision-making problems:
- [`DynamicVehicleSchedulingBenchmark`](@ref): multi-stage vehicle scheduling under customer uncertainty
- [`DynamicAssortmentBenchmark`](@ref): sequential product assortment selection with endogenous uncertainty
- [`MaintenanceBenchmark`](@ref): maintenance problem with resource constraint

## Getting Started

In a few lines of code, you can create benchmark instances, generate datasets, initialize learning components, and evaluate performance, using the same syntax across all benchmarks:

```julia
using DecisionFocusedLearningBenchmarks

# Create a benchmark instance for the argmax problem
benchmark = ArgmaxBenchmark()

# Generate training data
dataset = generate_dataset(benchmark, 100)

# Initialize policy components
model = generate_statistical_model(benchmark)
maximizer = generate_maximizer(benchmark)

# Training algorithm you want to use
# ... your training code here ...

# Evaluate performance
gap = compute_gap(benchmark, dataset, model, maximizer)
```

The only component you need to customize is the training algorithm itself.

## Related Packages

This package is part of the [JuliaDecisionFocusedLearning](https://github.com/JuliaDecisionFocusedLearning) organization, and built to be compatible with other packages in the ecosystem:
- **[InferOpt.jl](https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl)**: differentiable optimization layers and losses for decision-focused learning
- **[DecisionFocusedLearningAlgorithms.jl](https://github.com/JuliaDecisionFocusedLearning/DecisionFocusedLearningAlgorithms.jl)**: collection of generic black-box implementations of decision-focused learning algorithms
